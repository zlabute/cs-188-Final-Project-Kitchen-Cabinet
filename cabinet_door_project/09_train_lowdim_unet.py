"""
Step 9: Train a Diffusion UNet Low-Dim Policy
===============================================
Trains a 1-D ConditionalUnet diffusion policy on the augmented
low-dimensional dataset produced by 05b_augment_handle_data.py.

The policy learns to denoise action trajectories conditioned on a
short observation window (eef state + handle features = 13 dims),
producing temporally coherent 7-dim action *chunks* that are
executed open-loop before re-planning.

Usage:
    python 09_train_lowdim_unet.py
    python 09_train_lowdim_unet.py --config configs/diffusion_policy.yaml
    python 09_train_lowdim_unet.py --epochs 500 --batch_size 128
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import tqdm
import yaml

# Ensure local diffusion_policy package is importable
_DP_ROOT = os.path.join(os.path.dirname(__file__), "diffusion_policy")
if _DP_ROOT not in sys.path:
    sys.path.insert(0, _DP_ROOT)

from cabinet_utils import (
    HANDLE_OBS_COLS,
    HANDLE_XAXIS_KEEP,
    PARQUET_ROBOT_INDICES,
    ACTIVE_ACTION_INDICES,
    build_diffusion_policy,
    load_config,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply


# ===================================================================
# Dataset
# ===================================================================

def _get_dataset_path():
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def _find_augmented_dir(dataset_path: str) -> str:
    candidates = [
        os.path.join(dataset_path, "augmented"),
        os.path.join(dataset_path, "lerobot", "augmented"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "Augmented data not found. Run 05b_augment_handle_data.py first.\n"
        f"Searched: {candidates}"
    )


class CabinetAugmentedDataset(BaseLowdimDataset):
    """
    Sliding-window dataset over augmented parquet episodes.

    Each sample is a temporal window of length ``horizon``:
        {'obs': (horizon, 13), 'action': (horizon, 7)}

    Observations: eef_pos(3) + eef_quat(4) + handle_to_eef(3) + openness(1) + xaxis_xy(2).
    Actions: only the 7 active dims (original indices 5-11).
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        val_ratio: float = 0.02,
        max_episodes: int = None,
        is_validation: bool = False,
    ):
        super().__init__()
        import pyarrow.parquet as pq

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.pad_before = n_obs_steps - 1
        self.pad_after = n_action_steps - 1

        aug_dir = _find_augmented_dir(dataset_path)
        parquet_files = sorted(
            os.path.join(aug_dir, f) for f in os.listdir(aug_dir) if f.endswith(".parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(f"No augmented parquet files in {aug_dir}")

        if max_episodes is not None:
            parquet_files = parquet_files[:max_episodes]

        n_total = len(parquet_files)
        n_val = max(1, int(n_total * val_ratio))
        if is_validation:
            parquet_files = parquet_files[-n_val:]
        else:
            parquet_files = parquet_files[:-n_val] if n_val < n_total else parquet_files

        self._episode_obs = []
        self._episode_act = []
        self._indices = []

        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            obs, act = self._extract_episode(df)
            ep_idx = len(self._episode_obs)
            self._episode_obs.append(obs)
            self._episode_act.append(act)

            ep_len = obs.shape[0]
            for start in range(ep_len):
                self._indices.append((ep_idx, start))

        self.obs_dim = self._episode_obs[0].shape[1] if self._episode_obs else 13
        self.action_dim = self._episode_act[0].shape[1] if self._episode_act else 7

        self._val_ratio = val_ratio
        self._max_episodes = max_episodes
        self._dataset_path = dataset_path

    @staticmethod
    def _extract_episode(df):
        """Build (N, 13) obs and (N, 7) action arrays from a dataframe."""
        full_state = np.stack(df["observation.state"].values).astype(np.float32)
        robot_obs = full_state[:, PARQUET_ROBOT_INDICES]  # (N, 7)

        handle_parts = []
        for col in HANDLE_OBS_COLS:
            if col not in df.columns:
                continue
            arr = np.stack(df[col].values).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if col == "observation.handle_xaxis":
                arr = arr[:, :HANDLE_XAXIS_KEEP]
            handle_parts.append(arr)

        obs = np.concatenate([robot_obs] + handle_parts, axis=1)

        full_act = np.stack(df["action"].values).astype(np.float32)
        act = full_act[:, ACTIVE_ACTION_INDICES]  # (N, 7)

        return obs, act

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        ep_idx, start = self._indices[idx]
        obs_ep = self._episode_obs[ep_idx]
        act_ep = self._episode_act[ep_idx]
        ep_len = obs_ep.shape[0]

        window_start = start - self.pad_before
        window_end = window_start + self.horizon

        obs_window = np.zeros((self.horizon, self.obs_dim), dtype=np.float32)
        act_window = np.zeros((self.horizon, self.action_dim), dtype=np.float32)

        for i in range(self.horizon):
            t = window_start + i
            t_clamped = max(0, min(t, ep_len - 1))
            obs_window[i] = obs_ep[t_clamped]
            act_window[i] = act_ep[t_clamped]

        return {
            "obs": torch.from_numpy(obs_window),
            "action": torch.from_numpy(act_window),
        }

    def get_normalizer(self, mode="limits", **kwargs) -> LinearNormalizer:
        all_obs = np.concatenate(self._episode_obs, axis=0)
        all_act = np.concatenate(self._episode_act, axis=0)
        data = {
            "obs": torch.from_numpy(all_obs).float(),
            "action": torch.from_numpy(all_act).float(),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data, mode=mode, last_n_dims=1)
        return normalizer

    def get_validation_dataset(self) -> "CabinetAugmentedDataset":
        return CabinetAugmentedDataset(
            dataset_path=self._dataset_path,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            val_ratio=self._val_ratio,
            max_episodes=self._max_episodes,
            is_validation=True,
        )

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate(self._episode_act, axis=0)).float()


# ===================================================================
# Training loop
# ===================================================================

def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def train(cfg: dict):
    tcfg = cfg["training"]
    seed = tcfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ----
    print_section("Loading augmented dataset")
    dataset_path = _get_dataset_path()
    print(f"Dataset root: {dataset_path}")

    dcfg = cfg.get("dataset", {})
    dataset = CabinetAugmentedDataset(
        dataset_path=dataset_path,
        horizon=cfg["horizon"],
        n_obs_steps=cfg["n_obs_steps"],
        n_action_steps=cfg["n_action_steps"],
        val_ratio=dcfg.get("val_ratio", 0.02),
        max_episodes=dcfg.get("max_episodes", None),
    )
    val_dataset = dataset.get_validation_dataset()

    print(f"Train samples: {len(dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Obs dim:       {dataset.obs_dim}")
    print(f"Action dim:    {dataset.action_dim}")

    lcfg = cfg.get("dataloader", {})
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=lcfg.get("num_workers", 4),
        pin_memory=lcfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=lcfg.get("num_workers", 4),
        pin_memory=lcfg.get("pin_memory", True),
    )

    # ---- Policy ----
    print_section("Building diffusion policy")
    policy = build_diffusion_policy(cfg)

    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {n_params:,}")

    policy.to(device)

    # ---- EMA ----
    ema_policy = None
    if tcfg.get("use_ema", True):
        ema_policy = copy.deepcopy(policy)
        ema_policy.eval()
        ema_decay = tcfg.get("ema_decay", 0.995)
        print(f"EMA enabled (decay={ema_decay})")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg.get("weight_decay", 1e-6),
    )

    from diffusers.optimization import get_scheduler as _get_scheduler

    num_training_steps = (
        len(train_loader) * tcfg["num_epochs"]
    ) // tcfg.get("gradient_accumulate_every", 1)
    lr_scheduler = _get_scheduler(
        tcfg.get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=tcfg.get("lr_warmup_steps", 500),
        num_training_steps=num_training_steps,
    )

    # ---- Checkpointing ----
    ckpt_dir = cfg.get("checkpoint_dir", "tmp/cabinet_policy_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    grad_accum = tcfg.get("gradient_accumulate_every", 1)
    best_val_loss = float("inf")
    global_step = 0

    # ---- Training loop ----
    print_section("Training")
    print(f"Epochs:           {tcfg['num_epochs']}")
    print(f"Batch size:       {tcfg['batch_size']}")
    print(f"LR:               {tcfg['learning_rate']}")
    print(f"Gradient accum:   {grad_accum}")
    print(f"Checkpoint dir:   {ckpt_dir}")
    print()

    for epoch in range(tcfg["num_epochs"]):
        policy.train()
        train_losses = []

        pbar = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{tcfg['num_epochs']}",
            leave=False,
        )
        for batch_idx, batch in enumerate(pbar):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            raw_loss = policy.compute_loss(batch)
            loss = raw_loss / grad_accum
            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if ema_policy is not None:
                _ema_step(ema_policy, policy, ema_decay)

            train_losses.append(raw_loss.item())
            pbar.set_postfix(loss=raw_loss.item())
            global_step += 1

            max_steps = tcfg.get("max_train_steps")
            if max_steps is not None and batch_idx >= max_steps - 1:
                break

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")

        # ---- Validation ----
        val_loss = float("nan")
        if (epoch + 1) % tcfg.get("val_every", 50) == 0 and len(val_dataset) > 0:
            eval_policy = ema_policy if ema_policy is not None else policy
            eval_policy.eval()
            val_losses = []
            with torch.no_grad():
                for vb_idx, vbatch in enumerate(val_loader):
                    vbatch = dict_apply(vbatch, lambda x: x.to(device, non_blocking=True))
                    vl = eval_policy.compute_loss(vbatch)
                    val_losses.append(vl.item())
                    max_val = tcfg.get("max_val_steps")
                    if max_val is not None and vb_idx >= max_val - 1:
                        break
            if val_losses:
                val_loss = float(np.mean(val_losses))

        # ---- Logging ----
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = lr_scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch + 1:5d}  "
                f"train_loss={avg_train:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"lr={lr_now:.2e}"
            )

        # ---- Checkpointing ----
        save_policy = ema_policy if ema_policy is not None else policy
        ckpt_payload = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": avg_train,
            "val_loss": val_loss,
            "policy_state_dict": save_policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
        }

        if (epoch + 1) % tcfg.get("checkpoint_every", 100) == 0:
            path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1:05d}.pt")
            torch.save(ckpt_payload, path)

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_payload, os.path.join(ckpt_dir, "best_diffusion_policy.pt"))

    # ---- Final save ----
    final_payload = {
        "epoch": tcfg["num_epochs"],
        "global_step": global_step,
        "train_loss": avg_train,
        "val_loss": val_loss,
        "policy_state_dict": (ema_policy if ema_policy is not None else policy).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg,
    }
    torch.save(final_payload, os.path.join(ckpt_dir, "final_diffusion_policy.pt"))

    print_section("Training complete")
    print(f"  Best val loss:      {best_val_loss:.6f}")
    print(f"  Final checkpoint:   {os.path.join(ckpt_dir, 'final_diffusion_policy.pt')}")
    print(f"  Best checkpoint:    {os.path.join(ckpt_dir, 'best_diffusion_policy.pt')}")
    print(f"\n  Evaluate with:")
    print(f"    python 07_evaluate_policy.py --checkpoint {os.path.join(ckpt_dir, 'best_diffusion_policy.pt')}")


# ===================================================================
# EMA update (manual, avoids diffusers EMAModel dependency issues)
# ===================================================================

@torch.no_grad()
def _ema_step(ema_model, model, decay: float):
    for ep, mp in zip(ema_model.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(mp.data, alpha=1.0 - decay)
    for eb, mb in zip(ema_model.buffers(), model.buffers()):
        eb.data.copy_(mb.data)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a Diffusion UNet low-dim policy for OpenCabinet"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "diffusion_policy.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir

    print("=" * 60)
    print("  OpenCabinet - Diffusion UNet Low-Dim Training")
    print("=" * 60)

    train(cfg)


if __name__ == "__main__":
    main()
