import os
import random
import multiprocessing as mp
from typing import Tuple, Union, Optional, Any, Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class _NoOpLogger:
    def log_scalar(self, *args, **kwargs):
        pass


def _as_float_01_from_any(x: torch.Tensor) -> torch.Tensor:
    """
    Convert input images to float in [0,1].

    Accepts:
      - uint8 in [0,255]
      - float in [0,1]
      - float in [-0.5,0.5]
    """
    if x.dtype == torch.uint8:
        return x.float() / 255.0

    x = x.float()
    x_max = float(x.max().item()) if x.numel() else 0.0
    x_min = float(x.min().item()) if x.numel() else 0.0

    if x_max > 1.0:              # likely [0,255] but already float
        x = x / 255.0
        return x.clamp(0.0, 1.0)

    if x_min < 0.0:              # assume [-0.5,0.5]
        x = x + 0.5
        return x.clamp(0.0, 1.0)

    # already [0,1]
    return x.clamp(0.0, 1.0)


def workers_from_slurm(default_cap: int = 4) -> int:
    slurm = os.getenv("SLURM_CPUS_PER_TASK")
    n = int(slurm) if slurm and slurm.isdigit() else mp.cpu_count()
    return max(0, min(default_cap, n))


class AEImageDataset(Dataset):
    """
    Yields (C,H,W) float32 in [0,1].
    Accepts uint8 [0,255], float [0,1], or float [-0.5,0.5].
    """
    def __init__(self, images: Union[np.ndarray, torch.Tensor]):
        obs = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
        # Expect (N,C,H,W)
        if obs.ndim != 4:
            raise ValueError(f"Expected images (N,C,H,W), got shape {tuple(obs.shape)}")
        self.obs = _as_float_01_from_any(obs).contiguous()

    def __len__(self):
        return int(self.obs.shape[0])

    def __getitem__(self, idx):
        return self.obs[idx]


# Main model
class StateAutoEncoderModel(nn.Module):
    """
    CNN autoencoder for images (3,64,64).
    
    get_representation_torch(obs_bchw) -> (B, latent_dim) torch tensor on device
    reset/update/finalize_representation_stats_accum for rep-norm in p2
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
        default_num_workers_cap: int = 4,
        compile_encoder: bool = True,
        **_ignored,
    ):
        super().__init__()

        self._device = torch.device("cpu" if (not torch.cuda.is_available()) else device)
        self._seed = int(seed)
        self._normalize_representations = bool(normalize_representations)

        # Logging
        self._logger = logger if logger is not None else _NoOpLogger()

        # Threads/workers
        self._num_workers = workers_from_slurm(default_cap=default_num_workers_cap)
        torch.set_num_threads(max(1, self._num_workers))

        # Reproducible RNGs
        self._dataloader_generator = torch.Generator(device="cpu").manual_seed(self._seed)
        self._torch_gen = torch.Generator(device=self._device).manual_seed(self._seed + 1)

        # Shape
        C, H, W = in_dim
        assert C in (1, 3), f"Expected C=1 or 3, got {C}"
        assert (H, W) == (64, 64), f"Expected (3,64,64), got {(C,H,W)}"

        self._latent_dim = int(latent_dim)
        self._num_training_epochs = int(num_training_epochs)
        self._batch_size = int(batch_size)
        self._grad_clip = float(grad_clip)

        # Encoder: (B,C,64,64) -> (B, latent_dim)
        self._encoder_net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),   # 64 -> 15
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 48, kernel_size=4, stride=2),  # 15 -> 6
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # 6 -> 4
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B,64,1,1)
            nn.Flatten(),                 # -> (B,64)

            nn.LayerNorm(64),
            nn.Linear(64, self._latent_dim),
            nn.LayerNorm(self._latent_dim),
        )

        # Decoder: latent -> (B,C,64,64)
        self._decoder_seed = nn.Sequential(
            nn.Linear(self._latent_dim, 64),
            nn.ReLU(inplace=True),
        )

        self._decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 0),  # 1 -> 4
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 4 -> 8
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 8 -> 16
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 16 -> 32
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, C,  4, 2, 1),  # 32 -> 64
            nn.Sigmoid(),
        )

        # Move to device + channels_last for speed
        self._encoder_net = self._encoder_net.to(self._device)
        self._decoder_seed = self._decoder_seed.to(self._device)
        self._decoder_deconv = self._decoder_deconv.to(self._device)
        if self._device.type == "cuda":
            self._encoder_net.to(memory_format=torch.channels_last)
            self._decoder_deconv.to(memory_format=torch.channels_last)

        # Optimizer
        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        params = list(self._encoder_net.parameters()) + list(self._decoder_seed.parameters()) + list(self._decoder_deconv.parameters())
        self._optimizer = optimizer_fn(params)

        # Rep norm buffers
        self._repr_mean_t: Optional[torch.Tensor] = None
        self._repr_std_t: Optional[torch.Tensor] = None

        # Streaming accumulator (Welford)
        self._repr_stat_count = 0
        self._repr_stat_mean_t = None
        self._repr_stat_M2_t = None

        # Optional compile
        self._compile_encoder = bool(compile_encoder)
        self.enable_fast_inference(use_half=False, compile_encoder=self._compile_encoder)


    def _worker_init_fn(self):
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    
    def _to_01_bchw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to float32 in [0,1] and ensure BCHW.
        Accepts:
        - uint8 [0,255]
        - float [0,255]
        - float [0,1]
        - float [-0.5,0.5]
        Accepts HWC/CHW/BHWC/BCHW.
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        # Make BCHW
        if x.ndim == 3:  # CHW or HWC
            if x.shape[-1] in (1, 3):  # HWC
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)  # -> BCHW
        elif x.ndim == 4:  # BCHW or BHWC
            if x.shape[-1] in (1, 3):  # BHWC
                x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected 3D/4D image tensor, got {tuple(x.shape)}")

        # Convert to [0,1]
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
            return x.clamp_(0.0, 1.0)

        x = x.float()
        # Heuristics based on observed range
        x_min = float(x.min().item()) if x.numel() else 0.0
        x_max = float(x.max().item()) if x.numel() else 0.0

        if x_max > 1.5:               # float but looks like [0,255]
            x = x / 255.0
        elif x_min < -0.1:            # looks like [-0.5,0.5]
            x = x + 0.5               # -> [0,1] ideally

        return x.clamp(0.0, 1.0)


    # Data loader helpers

    def _get_buffer_data(self, buffer: Any) -> Dict[str, Any]:
        """
        Tries buffer.get_data(keys=[...]) first, falls back to buffer.get_data().
        """
        if buffer is None:
            raise ValueError("buffer is required")

        if hasattr(buffer, "get_data"):
            try:
                # preferred (like your NonlinearDynaQ version)
                return buffer.get_data(keys=["observation", "next_observation", "done", "terminal"])
            except TypeError:
                return buffer.get_data()
        raise TypeError("buffer must have get_data()")

    def prepare_train_loader(self, buffer: Any) -> DataLoader:
        data = self._get_buffer_data(buffer)

        # possible keys across your code versions
        obs = data.get("observation", None)
        next_obs = data.get("next_observation", None)

        if obs is None:
            raise ValueError("Buffer data missing 'observation' key")

        # Ensure numpy
        if torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        if torch.is_tensor(next_obs):
            next_obs = next_obs.detach().cpu().numpy()

        parts = [obs]
        if next_obs is not None and len(next_obs) > 0:
            parts.append(next_obs)

        images = np.concatenate(parts, axis=0)  # expects (N,C,H,W)

        dataloader_kwargs = dict(
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._dataloader_generator,
            num_workers=self._num_workers,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=(self._num_workers > 0),
            pin_memory=(self._device.type == "cuda"),
        )
        if self._num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 2

        return DataLoader(AEImageDataset(images), **dataloader_kwargs)


    # Loss
    def calculate_loss(
        self,
        x_bchw_01: torch.Tensor,
        lambda_binarize: float = 1e-3,
        noise_a: float = 0.3,
    ):
        """
        x_bchw_01: float in [0,1]
        """
        x = x_bchw_01.to(self._device).float()

        z = self._encoder_net(x)  # (B, D) logits
        b = torch.sigmoid(z)  # (B, D) in [0,1]

        # Anti-collapse terms to prevent all bits from going to 0 or 1 or being correlated or having 
        # low-entropy ()
        bit_mean = b.mean(dim=0).clamp(1e-3, 1 - 1e-3)
        balance_loss = ((bit_mean - 0.5) ** 2).mean()

        if b.shape[0] < 2:
            decorrel_loss = b.new_tensor(0.0)
        else:
            b0 = b - bit_mean
            cov = (b0.T @ b0) / (b.shape[0] - 1 + 1e-8)
            off_diag = cov.clone()
            off_diag.fill_diagonal_(0)
            decorrel_loss = (off_diag ** 2).mean()

        entropy_push = (b * (1 - b)).mean()

        lambda_bal = 1e-2
        lambda_dec = 1e-2
        lambda_ent = 1e-3
        anti_collapse = lambda_bal * balance_loss + lambda_dec * decorrel_loss + lambda_ent * entropy_push

        # Noise on b during training
        if self._encoder_net.training and noise_a is not None and noise_a > 0:
            noise = (2 * noise_a) * torch.rand(b.shape, device=b.device, generator=self._torch_gen) - noise_a
            b_tilde = torch.clamp(b + noise, 0.0, 1.0)
        else:
            b_tilde = b

        seed_vec = self._decoder_seed(b_tilde)  # (B,64)
        seed = seed_vec.view(-1, 64, 1, 1)  # (B,64,1,1)
        recon = self._decoder_deconv(seed)  # (B,C,H,W) in [0,1]

        recon_loss = F.binary_cross_entropy(recon.float(), x.float(), reduction="mean")

        dist0 = b.pow(2)
        dist1 = (1.0 - b).pow(2)
        binarize_loss = torch.minimum(dist0, dist1).mean()

        total = recon_loss + float(lambda_binarize) * binarize_loss + anti_collapse

        return (
            total,
            recon_loss.detach(),
            binarize_loss.detach(),
            anti_collapse.detach(),
            balance_loss.detach(),
            decorrel_loss.detach(),
            entropy_push.detach(),
        )


    # Training
    def train(self, buffer=None, num_epochs: Optional[int] = None, lambda_binarize: float = 1e-3, noise_a: float = 0.3):
        if buffer is None:
            raise ValueError("Pass replay buffer to AutoencoderStateHashModel.train(buffer=...)")

        if num_epochs is None:
            num_epochs = self._num_training_epochs

        loader = self.prepare_train_loader(buffer)

        self._encoder_net.train()
        self._decoder_seed.train()
        self._decoder_deconv.train()

        for _ in range(int(num_epochs)):
            running_total = running_recon = running_binar = 0.0
            running_anti = 0.0
            n_samples = 0

            for batch in loader:
                # Batch: (B,C,H,W) float in [0,1]
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(self._device, non_blocking=(self._device.type == "cuda")).float()

                self._optimizer.zero_grad(set_to_none=True)

                total, recon, binar, anti, bal, dec, ent = self.calculate_loss(
                    x, lambda_binarize=lambda_binarize, noise_a=noise_a
                )

                total.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(self._encoder_net.parameters(), self._grad_clip)
                self._optimizer.step()

                bs = int(x.shape[0])
                running_total += float(total.item()) * bs
                running_recon += float(recon.item()) * bs
                running_binar += float(binar.item()) * bs
                running_anti += float(anti.item()) * bs
                n_samples += bs

            den = max(n_samples, 1)
            self._logger.log_scalar("ae/total_loss", running_total / den, "agent")
            self._logger.log_scalar("ae/recon_loss", running_recon / den, "agent")
            self._logger.log_scalar("ae/binarize_loss", running_binar / den, "agent")
            self._logger.log_scalar("ae/anti_collapse", running_anti / den, "agent")

        self._encoder_net.eval()
        self._decoder_seed.eval()
        self._decoder_deconv.eval()
        self.enable_fast_inference(use_half=False, compile_encoder=self._compile_encoder)

        if self._normalize_representations:
            self.learn_representation_stats_from_buffer(buffer)


    # Fast inference/deploy
    def enable_fast_inference(self, use_half: bool = False, compile_encoder: bool = True):
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


    def has_rep_stats(self) -> bool:
        return (self._repr_mean_t is not None) and (self._repr_std_t is not None)


    @torch.no_grad()
    def get_representation_torch(self, obs: torch.Tensor) -> torch.Tensor:
        self._encoder_net.eval()
        x01 = self._to_01_bchw(obs).to(self._device, non_blocking=(self._device.type=="cuda"))
        z = self._encoder_net(x01)

        if self._normalize_representations:
            assert self.has_rep_stats(), "rep stats not set"
            z = (z - self._repr_mean_t) / self._repr_std_t
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)

        return z


    # Rep stats (streaming Welford)

    @torch.no_grad()
    def reset_representation_stats_accum(self) -> None:
        self._repr_stat_count = 0
        self._repr_stat_mean_t = None
        self._repr_stat_M2_t = None

    @torch.no_grad()
    def update_representation_stats_accum(self, obs: torch.Tensor) -> None:
        if not self._normalize_representations:
            return
        x01 = self._to_01_bchw(obs).to(self._device, non_blocking=(self._device.type=="cuda"))
        z = self._encoder_net(x01).detach()

        b = int(z.shape[0])
        if b == 0:
            return

        batch_mean = z.mean(dim=0)
        batch_var = z.var(dim=0, unbiased=False)

        if self._repr_stat_mean_t is None:
            self._repr_stat_mean_t = batch_mean
            self._repr_stat_M2_t = batch_var * b
            self._repr_stat_count = b
        else:
            count = int(self._repr_stat_count)
            new_count = count + b
            delta = batch_mean - self._repr_stat_mean_t
            self._repr_stat_mean_t = self._repr_stat_mean_t + delta * (b / new_count)
            self._repr_stat_M2_t = (
                self._repr_stat_M2_t
                + batch_var * b
                + (delta * delta) * (count * b / new_count)
            )
            self._repr_stat_count = new_count

    @torch.no_grad()
    def finalize_representation_stats_accum(self, clamp_std: float = 1e-3) -> None:
        if not self._normalize_representations:
            return
        if self._repr_stat_mean_t is None or int(self._repr_stat_count) <= 0:
            return

        var = self._repr_stat_M2_t / max(int(self._repr_stat_count), 1)
        std = torch.sqrt(var).clamp_min(float(clamp_std))
        mean = self._repr_stat_mean_t

        self._repr_mean_t = mean.to(self._device)
        self._repr_std_t = std.to(self._device)

    @torch.no_grad()
    def learn_representation_stats_from_buffer(self, buffer: Any, batch_size: int = 256) -> None:
        """
        Compute mean/std over encoder logits z from either:
        - a replay buffer (with get_data), or
        - a dict containing {"observation": ...}
        """
        if not self._normalize_representations:
            return

        # Accept dict directly (your phase2_data path)
        if isinstance(buffer, dict):
            data = buffer
        else:
            data = self._get_buffer_data(buffer)

        obs = data.get("observation", None)
        if obs is None:
            raise ValueError("Missing 'observation'")

        if torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()

        N = int(obs.shape[0])
        self.reset_representation_stats_accum()

        for s in range(0, N, int(batch_size)):
            batch_np = obs[s:s + int(batch_size)]
            batch = torch.as_tensor(batch_np, device=self._device)
            x01 = self._to_01_bchw(batch)  # Handles uint8/float/shapes
            self.update_representation_stats_accum(x01)
            
        self.finalize_representation_stats_accum()


    # Save/load

    def save(self, dname: str):
        os.makedirs(dname, exist_ok=True)
        torch.save(
            {
                "encoder": self._encoder_net.state_dict(),
                "decoder_seed": self._decoder_seed.state_dict(),
                "decoder_deconv": self._decoder_deconv.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "normalize_representations": self._normalize_representations,
                "repr_mean_t": self._repr_mean_t,
                "repr_std_t": self._repr_std_t,
                "seed": self._seed,
                "latent_dim": self._latent_dim,
            },
            os.path.join(dname, "ae_hash_model.pt"),
        )

    def load(self, dname: str, map_location=None):
        ckpt = torch.load(
            os.path.join(dname, "ae_hash_model.pt"),
            map_location=(map_location if map_location is not None else self._device),
        )
        self._encoder_net.load_state_dict(ckpt["encoder"])
        self._decoder_seed.load_state_dict(ckpt["decoder_seed"])
        self._decoder_deconv.load_state_dict(ckpt["decoder_deconv"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        self._normalize_representations = bool(ckpt.get("normalize_representations", self._normalize_representations))
        self._repr_mean_t = ckpt.get("repr_mean_t", None)
        self._repr_std_t = ckpt.get("repr_std_t", None)

        self._encoder_net.to(self._device).eval()
        self._decoder_seed.to(self._device).eval()
        self._decoder_deconv.to(self._device).eval()
        self.enable_fast_inference(use_half=False, compile_encoder=self._compile_encoder)