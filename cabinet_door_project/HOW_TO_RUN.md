# How to Run the Diffusion Policy Pipeline

This guide walks through every step from raw dataset to a trained, evaluated,
and visualized diffusion policy for the OpenCabinet task. All commands are run
from the `cabinet_door_project/` directory with the virtual environment active.

```bash
cd cabinet_door_project
source ../.venv/bin/activate
```

---

## Pipeline Overview

```
Step 0-4  (already done)       Environment setup, dataset download
    |
Step 5b   05b_augment_handle_data.py    Add handle features to parquet
    |
Step 9    09_train_lowdim_unet.py       Train the diffusion policy
    |
Step 7    07_evaluate_policy.py         Evaluate success rate
    |
Step 8    08_visualize_policy_rollout.py Watch the robot
```

---

## Step 1: Augment the Dataset (run once)

This replays the saved MuJoCo states from the demonstration dataset to extract
handle position, door openness, and other features the robot needs. It writes
augmented parquet files alongside the originals.

```bash
python 05b_augment_handle_data.py
```

Output: an `augmented/` directory inside the dataset folder containing one
parquet file per episode, each with the original columns plus 5 new
observation columns (11 extra dims total).

You only need to run this once. If you already have the augmented parquet
files, skip this step.

---

## Step 2: Train the Diffusion Policy

### Default (uses `configs/diffusion_policy.yaml`)

```bash
python 09_train_lowdim_unet.py
```

This trains for 3000 epochs with the settings in the YAML config. Checkpoints
are saved to `tmp/cabinet_policy_checkpoints/` by default.

### Override hyperparameters from the command line

```bash
# Shorter run for quick sanity check
python 09_train_lowdim_unet.py --epochs 100 --batch_size 64

# Custom learning rate
python 09_train_lowdim_unet.py --lr 3e-4

# Save checkpoints somewhere else
python 09_train_lowdim_unet.py --checkpoint_dir ./my_checkpoints

# Point to a different config file
python 09_train_lowdim_unet.py --config configs/diffusion_policy.yaml
```

### What gets saved

| File | Description |
|------|-------------|
| `best_diffusion_policy.pt` | Best checkpoint by validation loss |
| `final_diffusion_policy.pt` | Checkpoint at the end of training |
| `checkpoint_epoch_00100.pt` | Periodic checkpoints (every 100 epochs by default) |

Each checkpoint contains the full policy weights, normalizer state, optimizer
state, and the config dict so it is fully self-contained.

### Key config parameters (in `configs/diffusion_policy.yaml`)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `horizon` | 16 | Trajectory length the UNet denoises (timesteps) |
| `n_obs_steps` | 2 | Number of past observations used as context |
| `n_action_steps` | 8 | Number of actions executed before re-planning |
| `training.num_epochs` | 3000 | Total training epochs |
| `training.batch_size` | 256 | Batch size (reduce if you run out of memory) |
| `training.use_ema` | true | Exponential Moving Average of weights |
| `noise_scheduler.num_train_timesteps` | 100 | DDPM diffusion steps |

---

## Step 3: Evaluate the Policy

Run the trained policy in simulation and report success rate.

```bash
# Basic evaluation (20 episodes on pretrain kitchens)
python 07_evaluate_policy.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt

# More episodes for a reliable success rate
python 07_evaluate_policy.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt \
    --num_rollouts 50

# Evaluate on held-out kitchen scenes
python 07_evaluate_policy.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt \
    --split target

# Save a video of the evaluation
python 07_evaluate_policy.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt \
    --video_path tmp/eval_videos.mp4
```

### All evaluation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--num_rollouts` | 20 | Number of episodes to run |
| `--max_steps` | 500 | Max timesteps per episode |
| `--split` | `pretrain` | `pretrain` or `target` (held-out kitchens) |
| `--video_path` | None | If set, saves an MP4 of the evaluation |
| `--seed` | 0 | Random seed for environment layout |

---

## Step 4: Visualize the Policy

Watch the robot execute the policy in real time or save a video.

### On-screen (interactive viewer window)

```bash
# Linux / WSL
python 08_visualize_policy_rollout.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt

# macOS (must use mjpython for the viewer window)
mjpython 08_visualize_policy_rollout.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt
```

You can orbit the camera with the mouse while the robot runs.

### Off-screen (headless, saves video)

```bash
python 08_visualize_policy_rollout.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt \
    --offscreen

# Multiple episodes, custom output path
python 08_visualize_policy_rollout.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt \
    --offscreen --num_episodes 3 --video_path tmp/my_rollout.mp4
```

### All visualization flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `best_diffusion_policy.pt` | Path to `.pt` checkpoint |
| `--num_episodes` | 1 | Episodes to run |
| `--max_steps` | 300 | Max timesteps per episode |
| `--offscreen` | off | Render to video instead of viewer window |
| `--video_path` | `tmp/policy_rollout.mp4` | Output video path |
| `--fps` | 20 | Video frames per second |
| `--max_fr` | 20 | On-screen playback speed cap |
| `--seed` | 42 | Random seed |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: Augmented data not found` | Run `python 05b_augment_handle_data.py` first |
| `FileNotFoundError: Dataset not found` | Run `python 04_download_dataset.py` first |
| Out of GPU memory | Lower `training.batch_size` in `configs/diffusion_policy.yaml` (try 64 or 128) |
| Rendering crashes on Mac | Use `mjpython` instead of `python` for on-screen scripts |
| `GLFW error` on headless server | Set `export MUJOCO_GL=osmesa` before running |
| Training loss is NaN | Lower learning rate (try `--lr 5e-5`) |
| Very slow training on CPU | Expected. Use a GPU if available, or reduce `--epochs` |

---

## File Reference

| File | Purpose |
|------|---------|
| `cabinet_utils.py` | Shared utilities (handle extraction, policy construction, inference runner) |
| `configs/diffusion_policy.yaml` | All hyperparameters for training |
| `05b_augment_handle_data.py` | One-time data augmentation (handle + door features) |
| `09_train_lowdim_unet.py` | Training script |
| `07_evaluate_policy.py` | Quantitative evaluation (success rate) |
| `08_visualize_policy_rollout.py` | Visual debugging (viewer or video) |
