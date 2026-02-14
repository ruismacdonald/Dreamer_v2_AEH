import os
import random
import multiprocessing as mp
from typing import Tuple, Union, Optional, Any, Dict, Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class _NoOpLogger:
    def log_scalar(self, *args, **kwargs):
        pass


def workers_from_slurm(default_cap: int = 4) -> int:
    slurm = os.getenv("SLURM_CPUS_PER_TASK")
    n = int(slurm) if slurm and slurm.isdigit() else mp.cpu_count()
    return max(0, min(default_cap, n))


def _as_bchw_float01(images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert to BCHW float01
    Return BCHW float32 in [0,1].
    Accepts:
      - (N,C,H,W) uint8 in [0,255]
      - (N,C,H,W) float in [0,1] or [-0.5,0.5] or [0,255]
    """
    x = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
    if x.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {tuple(x.shape)}")

    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
        return x.clamp_(0.0, 1.0)

    x = x.float()
    x_min = float(x.min().item()) if x.numel() else 0.0
    x_max = float(x.max().item()) if x.numel() else 0.0

    if x_max > 1.5:       # looks like [0,255] float
        x = x / 255.0
    elif x_min < -0.1:    # looks like [-0.5,0.5]
        x = x + 0.5

    return x.clamp(0.0, 1.0)


class AEImageDataset(Dataset):
    """Yields CHW float32 in [0,1]."""
    def __init__(self, images_bchw: Union[np.ndarray, torch.Tensor]):
        x = _as_bchw_float01(images_bchw).contiguous()  # BCHW float01
        self.x = x

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx]  # CHW


class StateAutoEncoderModel(nn.Module):
    """
    Simple convolutional autoencoder for learning compact representations of image observations 
    for hashing.

    - Training batches are BCHW float32 in [0,1]
    - Encoder outputs:
        z_logits: (B, D) pre-sigmoid
        b_soft  : (B, D) post-sigmoid in [0,1]
    - Representation returned by get_representation():
        (D,) float32 for a single observation, optionally z-scored
    """

    def __init__(
        self,
        in_dim: Tuple[int, int, int],
        optimizer_fn=None,
        seed: int = 0,
        logger=None,
        latent_dim: int = 256,
        num_training_epochs: int = 5,
        batch_size: int = 32,
        device: str = "cuda",
        normalize_representations: bool = False,
        grad_clip: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        default_num_workers_cap: int = 4,
        compile_encoder: bool = True,
        **_ignored,
    ):
        super().__init__()

        self._device = torch.device("cpu" if (not torch.cuda.is_available()) else device)
        self._seed = int(seed)

        self._logger = logger if logger is not None else _NoOpLogger()

        self._latent_dim = int(latent_dim)
        self._num_training_epochs = int(num_training_epochs)
        self._batch_size = int(batch_size)
        self._grad_clip = float(grad_clip)

        self._normalize_representations = bool(normalize_representations)

        C, H, W = in_dim
        assert C in (1, 3), f"Expected C=1 or 3, got {C}"
        assert (H, W) == (64, 64), f"Expected (64,64), got {(H,W)}"

        # Dataloader workers
        self._num_workers = workers_from_slurm(default_cap=default_num_workers_cap)
        self._dataloader_generator = torch.Generator(device="cpu").manual_seed(self._seed)
        self._torch_gen = torch.Generator(device=self._device).manual_seed(self._seed + 1)

        # Encoder: x01 (B,C,64,64) -> z_logits (B,D)
        self._encoder_net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),  # 64 -> 15, downsample + extract low-level features (edges/colors)
            nn.ReLU(inplace=True),  # nonlinearity
            nn.Conv2d(32, 48, kernel_size=4, stride=2),  # 15 -> 6, deeper features + more downsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # 6 -> 4, refine spatial features at small resolution
            nn.ReLU(inplace=True),
            # Aaverages each channel over all HxW locations, so (B, C, H, W) -> (B, C, 1, 1). The model keeps 
            # what features exist but drops where exactly they are.
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B,64,1,1),  global pooling
            nn.Flatten(),  # (B,64,1,1) -> (B,64)
            nn.LayerNorm(64),  # Stabilize feature scale before projection
            nn.Linear(64, self._latent_dim),  # Project to latent logits (hash representation space)
            nn.LayerNorm(self._latent_dim),  # Stabilize latent scale across dims
        )

        # Decoder: b_soft -> recon
        self._decoder_seed = nn.Sequential(
            nn.Linear(self._latent_dim, 64),  # Map latent back to decoder channel space
            nn.ReLU(inplace=True),
        )
        self._decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 0),  # Upsample 1 -> 4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # Upsample 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # Upsample 8 -> 16, reduce channels to save compute
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # Upsample 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, C, 4, 2, 1),  # Upsample 32 -> 64, restore channels
            nn.Sigmoid(),  # Output in [0,1] for BCE reconstruction loss
        )

        self._encoder_net = self._encoder_net.to(self._device)
        self._decoder_seed = self._decoder_seed.to(self._device)
        self._decoder_deconv = self._decoder_deconv.to(self._device)

        if self._device.type == "cuda":
            self._encoder_net.to(memory_format=torch.channels_last)
            self._decoder_deconv.to(memory_format=torch.channels_last)

        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        params = list(self._encoder_net.parameters()) + list(self._decoder_seed.parameters()) + list(self._decoder_deconv.parameters())
        self._optimizer = optimizer_fn(params, lr=lr, weight_decay=weight_decay)

        # --- representation normalization state (torch tensors) ---
        # Store as buffers so they follow device + get saved in state_dict
        self.register_buffer("_repr_mean_t", torch.zeros(self._latent_dim, dtype=torch.float32))
        self.register_buffer("_repr_std_t", torch.ones(self._latent_dim, dtype=torch.float32))
        self._repr_stats_ready = False

        self._compile_encoder = bool(compile_encoder)
        self.enable_fast_inference(use_half=False, compile_encoder=self._compile_encoder)

    def _worker_init_fn(self, worker_id: int):
        # deterministic-ish workers
        seed = (self._seed + worker_id) % (2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


    # Data

    def _get_buffer_data(self, buffer: Any) -> Dict[str, Any]:
        if buffer is None:
            raise ValueError("buffer is required")
        if hasattr(buffer, "get_data"):
            try:
                return buffer.get_data(keys=["observation", "next_observation", "done", "terminal"])
            except TypeError:
                return buffer.get_data()
        raise TypeError("buffer must have get_data()")

    def prepare_train_loader(self, buffer: Any) -> DataLoader:
        data = self._get_buffer_data(buffer)
        obs = data.get("observation", None)
        next_obs = data.get("next_observation", None)

        if obs is None:
            raise ValueError("Buffer data missing 'observation'")

        if torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        if torch.is_tensor(next_obs):
            next_obs = next_obs.detach().cpu().numpy()

        parts = [obs]
        if next_obs is not None and len(next_obs) > 0:
            parts.append(next_obs)

        images = np.concatenate(parts, axis=0)  # (N,C,H,W) uint8 or float

        dl_kwargs = dict(
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._dataloader_generator,
            num_workers=self._num_workers,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=(self._num_workers > 0),
            pin_memory=(self._device.type == "cuda"),
        )
        if self._num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2

        return DataLoader(AEImageDataset(images), **dl_kwargs)


    # Core forward pieces

    def encode(self, x01_bchw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x01_bchw: BCHW float in [0,1]
        returns: (z_logits, b_soft)
        """
        x01 = x01_bchw.to(self._device)
        z_logits = self._encoder_net(x01)
        b_soft = torch.sigmoid(z_logits)
        return z_logits, b_soft

    def decode(self, b_soft_bD: torch.Tensor) -> torch.Tensor:
        seed_vec = self._decoder_seed(b_soft_bD)
        seed = seed_vec.view(-1, 64, 1, 1)
        recon = self._decoder_deconv(seed)
        return recon

    def calculate_loss(
        self,
        x01_bchw: torch.Tensor,
        lambda_binarize: float = 1e-3,
        noise_a: float = 0.3,
        lambda_bal: float = 1e-2,
        lambda_dec: float = 1e-2,
        lambda_ent: float = 1e-3,
    ):
        """
        x01_bchw: BCHW float32 in [0,1]
        """
        x01 = x01_bchw.to(self._device)

        z_logits, b_soft = self.encode(x01)

        # anti-collapse (on soft bits)
        bit_mean = b_soft.mean(dim=0).clamp(1e-3, 1 - 1e-3)
        balance_loss = ((bit_mean - 0.5) ** 2).mean()

        if b_soft.shape[0] < 2:
            decorrel_loss = b_soft.new_tensor(0.0)
        else:
            b0 = b_soft - bit_mean
            cov = (b0.T @ b0) / (b_soft.shape[0] - 1 + 1e-8)
            off = cov.clone()
            off.fill_diagonal_(0)
            decorrel_loss = (off ** 2).mean()

        entropy_push = (b_soft * (1 - b_soft)).mean()
        anti_collapse = lambda_bal * balance_loss + lambda_dec * decorrel_loss + lambda_ent * entropy_push

        # noise for robustness
        if self.training and noise_a and noise_a > 0:
            noise = (2 * noise_a) * torch.rand(b_soft.shape, device=b_soft.device, generator=self._torch_gen) - noise_a
            b_used = torch.clamp(b_soft + noise, 0.0, 1.0)
        else:
            b_used = b_soft

        recon = self.decode(b_used)

        recon_loss = F.binary_cross_entropy(recon, x01, reduction="mean")

        # Encourages bits to be close to 0 or 1 (soft binarization)
        dist0 = b_soft.pow(2)
        dist1 = (1.0 - b_soft).pow(2)
        binarize_loss = torch.minimum(dist0, dist1).mean()

        total = recon_loss + float(lambda_binarize) * binarize_loss + anti_collapse

        stats = {
            "ae_total": float(total.detach().cpu().item()),
            "ae_recon": float(recon_loss.detach().cpu().item()),
            "ae_binar": float(binarize_loss.detach().cpu().item()),
            "ae_anti": float(anti_collapse.detach().cpu().item()),
            "ae_bal": float(balance_loss.detach().cpu().item()),
            "ae_dec": float(decorrel_loss.detach().cpu().item()),
            "ae_ent": float(entropy_push.detach().cpu().item()),
            "ae_bit_mean_mean": float(bit_mean.mean().detach().cpu().item()),
            "ae_bit_mean_min": float(bit_mean.min().detach().cpu().item()),
            "ae_bit_mean_max": float(bit_mean.max().detach().cpu().item()),
        }

        return total, stats, recon.detach(), b_soft.detach(), z_logits.detach()

 
    # Training

    def train_on_buffer(
        self,
        buffer: Any,
        num_epochs: Optional[int] = None,
        lambda_binarize: float = 1e-3,
        noise_a: float = 0.3,
    ) -> Dict[str, float]:
        if num_epochs is None:
            num_epochs = self._num_training_epochs

        loader = self.prepare_train_loader(buffer)

        super().train(True)
        self._encoder_net.train()
        self._decoder_seed.train()
        self._decoder_deconv.train()

        last_epoch_stats = {}

        for epoch in range(int(num_epochs)):
            running = {}
            n = 0

            for x01_chw in loader:
                # Dataset yields CHW float01, dataloader stacks -> BCHW float01
                x01 = x01_chw.to(self._device, non_blocking=(self._device.type == "cuda"))

                self._optimizer.zero_grad(set_to_none=True)

                total, stats, _, _, _ = self.calculate_loss(
                    x01, lambda_binarize=lambda_binarize, noise_a=noise_a
                )

                total.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(self._encoder_net.parameters(), self._grad_clip)
                self._optimizer.step()

                bs = int(x01.shape[0])
                for k, v in stats.items():
                    running[k] = running.get(k, 0.0) + float(v) * bs
                n += bs

            den = max(n, 1)
            last_epoch_stats = {k: v / den for k, v in running.items()}
            for k, v in last_epoch_stats.items():
                self._logger.log_scalar(f"ae/{k}", v, "agent")

        # Switch to eval / fast inference
        self.eval()
        self.enable_fast_inference(use_half=False, compile_encoder=self._compile_encoder)

        # Compute and log rep stats
        data = self._get_buffer_data(buffer)
        stats = self.learn_representation_stats(data)
        for k, v in stats.items():
            self._logger.log_scalar(f"ae_stats/{k}", v, "agent")

        return last_epoch_stats


    # Representation stats + extraction

    @torch.no_grad()
    def learn_representation_stats(self, data: Dict[str, Any], batch_size: int = 256) -> Dict[str, float]:
        """
        Computes mean/std over logits across data["observation"].
        Stores as torch tensors on self._device.
        """
        obs = data.get("observation", None)
        if obs is None:
            raise ValueError("learn_representation_stats: data missing 'observation'")

        obs_bchw = _as_bchw_float01(obs)  # CPU BCHW float01

        reps = []
        N = int(obs_bchw.shape[0])
        for s in range(0, N, int(batch_size)):
            x01 = obs_bchw[s : s + int(batch_size)].to(self._device, non_blocking=(self._device.type == "cuda"))
            z_logits, b_soft = self.encode(x01)
            rep = z_logits
            reps.append(rep.float().cpu())

        R = torch.cat(reps, dim=0)  # CPU (N,D)
        mean = R.mean(dim=0)
        std = R.std(dim=0, unbiased=False).clamp_min(1e-8)

        if self._normalize_representations:
            self._repr_mean_t.copy_(mean.to(self._repr_mean_t.device))
            self._repr_std_t.copy_(std.to(self._repr_std_t.device))
            self._repr_stats_ready = True

        r_np = R.numpy()
        norms = np.linalg.norm(r_np, axis=1)

        out = {
            "rep_min": float(r_np.min()),
            "rep_max": float(r_np.max()),
            "rep_mean": float(r_np.mean()),
            "rep_std": float(r_np.std()),
            "rep_norm_mean": float(norms.mean()),
            "rep_norm_p99": float(np.quantile(norms, 0.99)),
            "repr_mean_abs_mean": float(mean.abs().mean().item()),
            "repr_std_mean": float(std.mean().item()),
            "repr_std_min": float(std.min().item()),
        }
        return out

    @torch.no_grad()
    def get_representation(self, obs) -> np.ndarray:
        """
        Accepts: CHW uint8, HWC uint8, BCHW, or BHWC.
        Returns: (D,) float32 rep.
        """
        self.eval()

        x = obs
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            x = torch.as_tensor(x)

        if x.ndim == 3:
            # If it's HWC, convert to CHW
            if x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)  # -> BCHW

        elif x.ndim == 4:
            # If it's BHWC, convert to BCHW
            if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected 3D/4D image, got {tuple(x.shape)}")

        x01 = _as_bchw_float01(x).to(self._device, non_blocking=(self._device.type == "cuda"))

        z_logits, _ = self.encode(x01)
        rep = z_logits  # (1,D)

        if self._normalize_representations:
            if not self._repr_stats_ready:
                raise RuntimeError("normalize_representations=True but stats not learned yet")
            rep = (rep - self._repr_mean_t) / self._repr_std_t

        return rep[0].detach().cpu().numpy().astype(np.float32, copy=False)


    # Fast inference / compile

    def enable_fast_inference(self, use_half: bool = False, compile_encoder: bool = True):
        self.eval()
        self._encoder_net.eval()
        if self._device.type == "cuda":
            self._encoder_net.to(memory_format=torch.channels_last)
            if use_half:
                self._encoder_net.half()

        if compile_encoder and self._device.type == "cuda":
            try:
                self._encoder_net = torch.compile(self._encoder_net, mode="reduce-overhead")
            except Exception:
                pass


    # Save / load

    def save(self, dname: str):
        os.makedirs(dname, exist_ok=True)
        torch.save(
            {
                "encoder": self._encoder_net.state_dict(),
                "decoder_seed": self._decoder_seed.state_dict(),
                "decoder_deconv": self._decoder_deconv.state_dict(),
                "optimizer": self._optimizer.state_dict(),

                "seed": self._seed,
                "latent_dim": self._latent_dim,
                "normalize_representations": self._normalize_representations,

                "repr_mean_t": self._repr_mean_t.detach().cpu(),
                "repr_std_t": self._repr_std_t.detach().cpu(),
                "repr_stats_ready": self._repr_stats_ready,
            },
            os.path.join(dname, "ae_hash_model.pt"),
        )

    def load(self, dname: str, map_location=None):
        ckpt = torch.load(
            os.path.join(dname, "ae_hash_model.pt"),
            map_location=(map_location if map_location is not None else "cpu"),
        )
        self._encoder_net.load_state_dict(ckpt["encoder"])
        self._decoder_seed.load_state_dict(ckpt["decoder_seed"])
        self._decoder_deconv.load_state_dict(ckpt["decoder_deconv"])
        self._optimizer.load_state_dict(ckpt["optimizer"])

        self._normalize_representations = bool(ckpt.get("normalize_representations", self._normalize_representations))

        mean = ckpt.get("repr_mean_t", None)
        std = ckpt.get("repr_std_t", None)
        if mean is not None and std is not None:
            self._repr_mean_t.copy_(mean.to(self._repr_mean_t.device))
            self._repr_std_t.copy_(std.to(self._repr_std_t.device))
            self._repr_stats_ready = bool(ckpt.get("repr_stats_ready", True))

        self._encoder_net.to(self._device).eval()
        self._decoder_seed.to(self._device).eval()
        self._decoder_deconv.to(self._device).eval()
        self.enable_fast_inference(use_half=False, compile_encoder=self._compile_encoder)