import torch
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Two modes:
        1. Standard circular replay buffer (ae_process=False)
            - Stores transitions in a single global ring buffer.
            - sample() returns Dreamer-style temporal sequences of length `seq_len`.
        2. SimHash-indexed local-forgetting (ae_process=True)
            - Still stores transitions in the SAME global ring buffer (time-ordered).
            - Additionally maintains per-hash FIFO queues of *global indices* to decide
                which transitions are "kept" vs "discarded" (local forgetting).

    SimHash key from the state-ae representation passed to add():
        dots = rep @ A_latent.T  # (hash_bits,)
        bits = (dots >= 0).astype(np.uint8)  # (hash_bits,)
        packed = np.packbits(bits, bitorder=...)  # bytes key
        key = bytes(packed)

    Local forgetting:
    - Every new transition is written to the global ring and gets reward_mask[idx] = 1.
    - For each hash key, keep at most `obs_hash_count` indices in a FIFO.
    - If inserting a new index makes that FIFO exceed capacity, evict the oldest LIVE
        index from that FIFO and set reward_mask[evicted_idx] = 0 (discarded).
    - When a discarded ring slot is later overwritten by a new transition, its
        reward_mask is set back to 1 as usual.

    Sampling:
    - Choose the START index uniformly from the set of currently-kept indices
        (loca_indices_flat), which guarantees reward_mask[start] == 1.
    - Return a temporal sequence from the global ring:
            idxs = [start, start+1, ..., start+seq_len-1] (mod buffer size)
    - The sequence may cross hash regions naturally; hashing only controls which
        start indices remain eligible.
    """

    def __init__(
        self,
        size,
        obs_shape,
        action_size,
        seq_len,
        batch_size,
        ae_process: bool = False,
        obs_hash_size: int = 32,
        obs_hash_count: int = 2000,
        seed: int = 0,
        packbit_order: str = "little",
    ):
        self.size = int(size)
        self.obs_shape = tuple(obs_shape)
        self.action_size = int(action_size)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)

        self.ae_process = bool(ae_process)

        # Stats
        self.steps, self.episodes = 0, 0

        # Standard ring-buffer state
        self.idx = 0
        self.full = False
        self.observations = np.empty((self.size, *self.obs_shape), dtype=np.uint8)
        self.actions = np.empty((self.size, self.action_size), dtype=np.float32)
        self.rewards = np.empty((self.size,), dtype=np.float32)
        self.terminals = np.empty((self.size,), dtype=np.float32)
        self.reward_mask = np.zeros((self.size,), dtype=np.float32)
        
        self.loca_indices = {}  # dict[hash -> deque[(idx, insert_id)]], FIFO per hash
        self.loca_indices_flat = []  # list of currently-kept idx (unique)
        self.flat_pos = np.full(self.size, -1, dtype=np.int64)  # idx -> position in loca_indices_flat
        self.insert_id = np.zeros(self.size, dtype=np.int64)  # generation stamp per slot
        self._global_insert_id = 0

        # SimHash-bucketed state
        if self.ae_process:
            self.hash_bits = int(obs_hash_size)  # SimHash bits
            self.fifo_capacity = int(obs_hash_count)  # FIFO capacity per hash

            if self.hash_bits <= 0:
                raise ValueError(f"obs_hash_size must be > 0, got {self.hash_bits}")
            if self.fifo_capacity <= 0:
                raise ValueError(f"obs_hash_count must be > 0, got {self.fifo_capacity}")

            if packbit_order not in ("little", "big"):
                raise ValueError(f"packbit_order must be 'little' or 'big', got {packbit_order}")
            self.packbit_order = packbit_order

            self._seed = int(seed)

            # Representation dim is unknown until we see the first representation
            self.obs_repr_size = None  # inferred later
            self.A_latent = None  # (hash_bits, obs_repr_size) created lazily

    def _flat_add(self, idx: int) -> None:
        """Add idx to the flat kept-set if not present."""
        if self.flat_pos[idx] != -1:
            return
        self.flat_pos[idx] = len(self.loca_indices_flat)
        self.loca_indices_flat.append(idx)

    def _flat_remove(self, idx: int) -> None:
        """Remove idx from the flat kept-set in O(1) (swap-delete)."""
        pos = int(self.flat_pos[idx])
        if pos == -1:
            return
        last_idx = self.loca_indices_flat[-1]
        self.loca_indices_flat[pos] = last_idx
        self.flat_pos[last_idx] = pos
        self.loca_indices_flat.pop()
        self.flat_pos[idx] = -1

    # SimHash helpers
    def _ensure_simhash_matrix(self, rep: np.ndarray) -> None:
        """Initialize A_latent once we know the representation dim"""
        rep = np.asarray(rep, dtype=np.float32).reshape(-1)

        if self.obs_repr_size is None:
            self.obs_repr_size = int(rep.shape[0])
            rng = np.random.default_rng(self._seed)
            A = rng.standard_normal(size=(self.hash_bits, self.obs_repr_size)).astype(np.float32)

            norms = np.linalg.norm(A, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # divide-by-zero guard, if row of A is zero norm (all 0s) set to 1 (stays all 0s after division)
            self.A_latent = A / norms
            
    def _simhash_key(self, rep: np.ndarray) -> bytes:
        """
        rep: shape (obs_repr_size,)

        Returns: bytes (packed bits) to use as dict key.
        """
        rep = np.asarray(rep, dtype=np.float32).reshape(-1)
        self._ensure_simhash_matrix(rep)

        # Now rep dim matches A_latent by construction
        dots = rep @ self.A_latent.T  # (hash_bits,)
        bits = (dots >= 0).astype(np.uint8)  # (hash_bits,)
        packed = np.packbits(bits, bitorder=self.packbit_order)
        return packed.tobytes()

    def _get_fifo(self, key: bytes) -> deque:
        """
        Return the per-hash FIFO for this key.

        Each FIFO stores tuples (idx, insert_id) where:
        - idx is the global ring slot index
        - insert_id is the generation stamp for that slot at insertion time
        """
        fifo = self.loca_indices.get(key)
        if fifo is None:
            fifo = deque()  # capacity enforced manually so we can handle stale entries cleanly
            self.loca_indices[key] = fifo
        return fifo

    def add(self, obs, action, reward, done, rep=None):
        """
        observation: dict with key "image"
        action: action vector
        reward: float
        done: bool/float
        representation: torch.Tensor or np.ndarray, required if ae_process=True
        """
        i = self.idx

        # Write to global ring (always)
        self.observations[i] = obs["image"]
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminals[i] = done
        self.reward_mask[i] = 1.0  # temporally overwritten (circular) slots become valid again

        if self.ae_process:
            if rep is None:
                raise ValueError("ae_process=True requires `representation` in add().")

            if isinstance(rep, torch.Tensor):
                rep = rep.detach().cpu().numpy().astype(np.float32).reshape(-1)
            else:
                rep = np.asarray(rep, dtype=np.float32).reshape(-1)

            # Stamp this slot-version
            self._global_insert_id += 1
            self.insert_id[i] = self._global_insert_id

            key = self._simhash_key(rep)
            fifo = self._get_fifo(key)

            # Ensure this new idx is eligible as a START index
            self._flat_add(i)

            # Add (idx, insert_id) so old references become stale on overwrite
            fifo.append((i, self.insert_id[i]))

            # If bucket too large, discard oldest LIVE entry
            while len(fifo) > self.fifo_capacity:
                disc_idx, disc_ins_idx = fifo.popleft()
                # stale? (slot reused since it was bucketed) -> ignore
                if self.insert_id[disc_idx] != disc_ins_idx:
                    continue
                # live eviction: mark discarded + remove from start pool
                if self.reward_mask[disc_idx] != 0.0:
                    self.reward_mask[disc_idx] = 0.0
                    self._flat_remove(disc_idx)
                break

        # Advance ring pointer + stats
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        self.episodes += (1 if done else 0)

    # Sampling

    def _sample_idx(self, L):
        """Standard Dreamer sampling from the global ring."""
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = self.idx not in idxs[1:]
        return idxs

    def _sample_idx_ae(self, L: int):
        if len(self.loca_indices_flat) == 0:
            raise RuntimeError("No kept indices available yet (loca_indices_flat is empty).")

        valid_idx = False
        while not valid_idx:
            start = int(np.random.choice(self.loca_indices_flat))
            idxs = (np.arange(start, start + L) % self.size).astype(np.int64)
            valid_idx = self.idx not in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = self.observations[vec_idxs]
        return (
            observations.reshape(L, n, *observations.shape[1:]),
            self.actions[vec_idxs].reshape(L, n, -1),
            self.rewards[vec_idxs].reshape(L, n),
            self.terminals[vec_idxs].reshape(L, n),
            self.reward_mask[vec_idxs].reshape(L, n),
        )

    def sample(self):
        if self.ae_process and not self.full and self.idx < self.seq_len:
            raise RuntimeError(f"Sampling too early: idx={self.idx}, seq_len={self.seq_len}")
        
        n = self.batch_size
        L = self.seq_len

        if not self.ae_process:
            idxs = np.asarray([self._sample_idx(L) for _ in range(n)])
        else:
            idxs = np.asarray([self._sample_idx_ae(L) for _ in range(n)])

        obs, acs, rews, terms, reward_mask = self._retrieve_batch(idxs, n, L)
        return obs, acs, rews, terms, reward_mask

    def report_statistics(self):
        if not self.ae_process:
            return {
                "rewards_statistics": [
                    (float(r), int(cnt))
                    for r, cnt in zip(*np.unique(self.rewards, return_counts=True))
                ]
            }

        # ae_process=True
        fifo_sizes = [len(dq) for dq in self.loca_indices.values()]
        total = int(np.sum(fifo_sizes)) if fifo_sizes else 0
        return {
            "num_fifos": len(self.loca_indices),
            "kept_starts": len(self.loca_indices_flat),
            "total_indices_in_fifos": total,
            "fifo_size_statistics": {
                "min": int(np.min(fifo_sizes)) if fifo_sizes else 0,
                "max": int(np.max(fifo_sizes)) if fifo_sizes else 0,
                "mean": float(np.mean(fifo_sizes)) if fifo_sizes else 0.0,
            },
        }

    def get_data(self):
        data = {
            # Keep as uint8, no torch, no float32, no normalization here
            "observation": self.observations[: self.idx].copy(),  # uint8
            "terminal": self.terminals[: self.idx].copy(),
        }
        if self.ae_process:
            data.update({"loca_indices_flat": self.loca_indices_flat.copy()})
        return data