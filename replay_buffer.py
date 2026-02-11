import torch
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    SimHash-indexed local-forgetting of ae encoded states
        - Still stores transitions in a single global circular buffer (time-ordered).
            - sample() returns Dreamer-style temporal sequences of length `seq_len`.
        - Additionally maintains per-hash FIFO queues of *global indices* to decide
            which transitions are "kept" vs "discarded" (local forgetting).

    SimHash key from the state ae representation passed to add():
        dots = rep @ A_latent_t.T  # (hash_bits,)
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
        ae_process: bool = True,
        obs_hash_size: int = 32,
        obs_hash_count: int = 2000,
        seed: int = 0,
        packbit_order: str = "little",
    ):
        self._rng = np.random.default_rng(seed)
        self.ae_process = bool(ae_process)
        self.size = int(size)
        self.obs_shape = tuple(obs_shape)
        self.action_size = int(action_size)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)

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

    def _invalidate_slot_membership(self, i: int):
        if self.flat_pos[i] != -1:
            self._flat_remove(i)

    # SimHash helpers
    def _ensure_simhash_matrix(self, rep_t: torch.Tensor) -> None:
        """Initialize A_latent_t once we know the representation dim"""
        # rep_t: (B, D) or (D,)
        D = int(rep_t.shape[-1])
        if self.obs_repr_size is None:
            self.obs_repr_size = D

            rng = np.random.default_rng(self._seed)
            A = rng.standard_normal(size=(self.hash_bits, D)).astype(np.float32)
            A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)

            # store torch copy on the device where reps live
            self.A_latent_t = torch.as_tensor(A, device=rep_t.device, dtype=torch.float32)
            
    def _simhash_key_u32(self, rep_t: torch.Tensor, A_t: torch.Tensor) -> torch.Tensor:
        """
        u32 = unsigned 32-bit integer
        We 32 hash bits (hash_bits = 32, those 32 bits can be packed into a single integer in 
        the range 0 to 2^32 − 1.
        key_u32 = the SimHash key stored as one 32-bit unsigned int (is tiny bc 1 number, is much faster)

        rep_t: (B, D) float32 on GPU
        A_t: (32, D) float32 on GPU, row-normalized
        returns: (B,) torch.int64 keys (each fits in 32 bits), bytes (packed bits) to use as dict key.
        """
        dots = rep_t @ A_t.T  # (B, 32)
        bits = (dots >= 0).to(torch.int64)  # (B, 32) in {0,1}

        # bit 0 = least significant bit (LSB) (matches "little" convention)
        assert self.hash_bits == 32
        weights = (1 << torch.arange(32, device=rep_t.device, dtype=torch.int64))  # (32,)
        keys = (bits * weights).sum(dim=-1)  # (B,) int64, values < 2^32
        return keys

    def _get_fifo(self, key_u32: int) -> deque:
        """
        Return the per-hash FIFO for this key.

        Each FIFO stores tuples (idx, insert_id) where:
        - idx is the global ring slot index
        - insert_id is the generation stamp for that slot at insertion time
        """
        fifo = self.loca_indices.get(key_u32)
        if fifo is None:
            fifo = deque()  # capacity enforced manually so we can handle stale entries cleanly
            self.loca_indices[key_u32] = fifo
        return fifo

    def add(self, obs, action, reward, done, rep_t=None, key_u32=None):
        """
        observation: dict with key "image"
        action: action vector
        reward: float
        done: bool/float
        representation: torch.Tensor or np.ndarray
        """
        i = self.idx
        # Overwriting slot i due to circular replay: remove stale kept-start membership; we'll 
        # re-add for the new transition (kept-start mirrors reward_mask).
        self._invalidate_slot_membership(i)

        # Write to global ring (always)
        self.observations[i] = obs["image"]
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminals[i] = done
        self.reward_mask[i] = 1.0  # Temporally overwritten (circular) slots become valid again

        if not self.ae_process:
            # Just advance ring
            self.idx = (self.idx + 1) % self.size
            self.full = self.full or self.idx == 0
            self.steps += 1
            self.episodes += (1 if done else 0)
            return

        if key_u32 is None:
            if rep_t is None:
                raise ValueError("Need rep_t or key_u32")

            # rep_t should be torch on GPU (D,) or (1,D)
            if rep_t.ndim == 1:
                rep_t = rep_t.unsqueeze(0)  # (1,D)

            self._ensure_simhash_matrix(rep_t)
            key_t = self._simhash_key_u32(rep_t, self.A_latent_t)  # (1,)
            key_u32 = int(key_t[0].item())  # tiny CPU transfer (one int)

        # Stamp this slot-version
        self._global_insert_id += 1
        self.insert_id[i] = self._global_insert_id

        fifo = self._get_fifo(key_u32)

        # Ensure this new idx is eligible as a start index
        self._flat_add(i)

        # Add (idx, insert_id) so old references become stale on overwrite
        fifo.append((i, self.insert_id[i]))

        # If bucket too large, discard oldest valid (still-current) entry
        while len(fifo) > self.fifo_capacity:
            disc_idx, disc_ins_idx = fifo.popleft()
            # stale? (slot reused since it was bucketed) -> ignore
            if self.insert_id[disc_idx] != disc_ins_idx:
                continue
            # valid eviction: mark discarded + remove from start pool
            if self.reward_mask[disc_idx] != 0.0:
                self.reward_mask[disc_idx] = 0.0
                self._flat_remove(disc_idx)

        # Advance ring pointer + stats
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        self.episodes += (1 if done else 0)

    # Sampling
    
    def _sample_idx(self, L: int):
        """Choose start index and return a temporal window of length L."""
        if not self.ae_process:
            # Sample start uniformly from the *valid* portion of the ring
            if not self.full:
                max_start = self.idx - L
                if max_start < 0:
                    raise RuntimeError("Not enough data to sample yet.")
                start = int(self._rng.integers(0, max_start + 1))
                return np.arange(start, start + L, dtype=np.int64)
            else:
                # Full ring: avoid windows that cross the current write index
                while True:
                    start = int(self._rng.integers(0, self.size))
                    idxs = (np.arange(start, start + L) % self.size).astype(np.int64)
                    if self.idx in idxs[1:]:
                        continue
                    return idxs
        
        if len(self.loca_indices_flat) == 0:
            raise RuntimeError("No kept indices available yet (loca_indices_flat is empty).")

        while True:
            start = int(self._rng.choice(self.loca_indices_flat))

            # if not full, require contiguous window entirely inside [0, self.idx)
            if not self.full:
                if start + L > self.idx:
                    continue
                idxs = np.arange(start, start + L, dtype=np.int64)
            else:
                idxs = (np.arange(start, start + L) % self.size).astype(np.int64)
                if self.idx in idxs[1:]:
                    continue

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
        n = self.batch_size
        L = self.seq_len

        idxs = np.asarray([self._sample_idx(L) for _ in range(n)])

        obs, acs, rews, terms, reward_mask = self._retrieve_batch(idxs, n, L)
        return obs, acs, rews, terms, reward_mask

    def report_statistics(self):
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
            "reward_mask_sum_global": float(self.reward_mask.sum()),
            "reward_mask_mean_global": float(self.reward_mask.mean()),
        }

    def get_data(self):
        observations = self.observations[: self.idx].copy()  # uint8 (N,C,H,W)
        return {
            "observation": observations,
            "terminal": self.terminals[: self.idx].copy(),
            "loca_indices_flat": self.loca_indices_flat.copy(),
        }