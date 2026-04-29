# Kitchen Cabinet Door Opening — Robotic Manipulation with Behavior Cloning & Diffusion Policy

A learning-from-demonstration pipeline for teaching a 7-DOF mobile manipulator to open kitchen cabinet doors. The project spans data collection via teleoperation, low-dimensional state augmentation, and two policy architectures: an MLP behavior-cloning baseline and a ConditionalUnet1D diffusion policy.

Built on top of [RoboCasa](https://robocasa.ai/) and [robosuite](https://robosuite.ai/), running in [MuJoCo 3.3.1](https://mujoco.readthedocs.io/) physics simulation across 2,500+ procedurally varied kitchen scenes.

---

## Highlights

- **Task**: `OpenCabinet` — robot must grasp and pull open a hinged cabinet door from a closed position
- **Robot**: PandaOmron — Franka Panda 7-DOF arm on an Omron wheeled base with torso lift
- **Policies**: MLP regression baseline + ConditionalUnet1D diffusion policy with DDPM noise scheduling
- **Data**: RoboCasa pre-collected demonstrations (human + MimicGen-expanded), augmented with handle position and door-openness features extracted from replay
- **Dataset format**: [LeRobot](https://github.com/huggingface/lerobot) (parquet + video)
- **Eval**: Success rate over N rollouts, with pretrain/target kitchen splits for generalization testing

---

## Installation

Run the install script (macOS and WSL/Linux):

```bash
./install.sh
```

This will:
- Create a Python virtual environment (`.venv`)
- Clone and install robosuite and robocasa
- Install all Python dependencies (PyTorch, numpy, matplotlib, etc.)
- Download RoboCasa kitchen assets (~10 GB)

Activate the environment after installation:

```bash
source .venv/bin/activate
```

Verify everything is working:

```bash
cd cabinet_door_project
python 00_verify_installation.py
```

> **macOS note:** Scripts that open a rendering window (03, 05, 08) require `mjpython` instead of `python`.

---

## Project Structure
cabinet_door_project/
00_verify_installation.py        # Check MuJoCo, robosuite, RoboCasa, dependencies
01_explore_environment.py        # Inspect observation/action spaces
02_random_rollouts.py            # Sanity-check random policy, save video
03_teleop_collect_demos.py       # Keyboard teleoperation for demo collection
04_download_dataset.py           # Download official OpenCabinet dataset
05_playback_demonstrations.py    # Visualize expert demonstrations
05b_augment_handle_data.py       # Add handle position & door features to dataset
06_train_policy.py               # MLP behavior-cloning baseline
07_evaluate_policy.py            # Quantitative evaluation (success rate)
08_visualize_policy_rollout.py   # Interactive viewer or video export
09_train_lowdim_unet.py          # ConditionalUnet1D diffusion policy training
cabinet_utils.py                 # Shared utilities
configs/
diffusion_policy.yaml          # Diffusion policy hyperparameters
figures/                         # Generated plots
plot_generators/                 # Scripts to regenerate figures
notebook.ipynb                   # Jupyter notebook companion
install.sh
README.md

---

## Pipeline

### 1 — Explore the environment

```bash
python 01_explore_environment.py
```

Prints the full observation and action space, task description, and success criteria for the `OpenCabinet` environment.

### 2 — Collect demonstrations (optional)

```bash
# macOS: mjpython
python 03_teleop_collect_demos.py
```

Control the robot with a keyboard to generate your own demonstrations. Good for building intuition about task difficulty before working with the pre-collected dataset.

**Keyboard controls:**

| Key | Action |
|-----|--------|
| Ctrl+q | Reset simulation |
| spacebar | Toggle gripper |
| ↑ ↓ ← → | Move in x-y plane |
| . ; | Move vertically |
| o p | Yaw |
| y h | Pitch |
| e r | Roll |
| b | Toggle arm/base control |

### 3 — Download the dataset

```bash
python 04_download_dataset.py
```

Downloads the official OpenCabinet demonstrations (human + MimicGen-expanded) from RoboCasa servers, stored in LeRobot format.

### 4 — Augment with handle features

```bash
python 05b_augment_handle_data.py
```

Replays saved MuJoCo states to extract handle position, door openness, and related geometry features. Writes augmented parquet files (11 extra observation dims) to an `augmented/` directory alongside the originals. Only needs to be run once.

### 5 — Train

**MLP baseline:**
```bash
python 06_train_policy.py
```

**Diffusion policy:**
```bash
python 09_train_lowdim_unet.py

# Override hyperparameters
python 09_train_lowdim_unet.py --epochs 100 --batch_size 64 --lr 3e-4
```

Checkpoints saved:

| File | Description |
|------|-------------|
| `best_diffusion_policy.pt` | Best by validation loss |
| `final_diffusion_policy.pt` | End-of-training checkpoint |
| `checkpoint_epoch_XXXXX.pt` | Periodic checkpoints |

Each checkpoint is self-contained: weights, normalizer state, optimizer state, and config.

### 6 — Evaluate

```bash
python 07_evaluate_policy.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt \
    --num_rollouts 50 \
    --split pretrain   # or 'target' for held-out kitchens
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--num_rollouts` | 20 | Episodes to run |
| `--max_steps` | 500 | Timesteps per episode |
| `--split` | `pretrain` | `pretrain` or `target` |
| `--video_path` | None | Save evaluation video |
| `--seed` | 0 | Random seed |

### 7 — Visualize

```bash
# Interactive viewer (macOS: use mjpython)
mjpython 08_visualize_policy_rollout.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt

# Headless, save video
python 08_visualize_policy_rollout.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt \
    --offscreen --video_path tmp/rollout.mp4
```

---

## Environment & Architecture Details

### Task

| Property | Value |
|----------|-------|
| Goal | Open a hinged cabinet door |
| Fixture | `HingeCabinet` |
| Initial state | Door closed, robot positioned nearby |
| Success | `fixture.is_open(env) == True` |
| Horizon | 500 steps at 20 Hz (25 s) |
| Scene variety | 2,500+ kitchen layouts |

### Observation Space (PandaOmron)

| Key | Shape | Description |
|-----|-------|-------------|
| `robot0_agentview_left_image` | (256, 256, 3) | Left shoulder camera |
| `robot0_agentview_right_image` | (256, 256, 3) | Right shoulder camera |
| `robot0_eye_in_hand_image` | (256, 256, 3) | Wrist camera |
| `robot0_gripper_qpos` | (2,) | Finger positions |
| `robot0_base_pos` | (3,) | Base position |
| `robot0_base_quat` | (4,) | Base orientation |
| `robot0_base_to_eef_pos` | (3,) | EEF position relative to base |
| `robot0_base_to_eef_quat` | (4,) | EEF orientation relative to base |

### Action Space

| Key | Dim | Description |
|-----|-----|-------------|
| `end_effector_position` | 3 | Delta (dx, dy, dz) |
| `end_effector_rotation` | 3 | Delta rotation (axis-angle) |
| `gripper_close` | 1 | 0 = open, 1 = close |
| `base_motion` | 4 | (forward, side, yaw, torso) |
| `control_mode` | 1 | 0 = arm, 1 = base |

### Diffusion Policy Config (`configs/diffusion_policy.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 16 | Trajectory length to denoise |
| `n_obs_steps` | 2 | Past observations used as context |
| `n_action_steps` | 8 | Actions executed before re-planning |
| `training.num_epochs` | 100 | Training epochs |
| `training.batch_size` | 64 | Batch size |
| `training.learning_rate` | 1e-4 | Learning rate (cosine decay) |
| `training.use_ema` | true | Exponential Moving Average |
| `noise_scheduler.num_train_timesteps` | 100 | DDPM diffusion steps |

### Simulation Stack
Kitchen Scene (2500+ layouts)    OpenCabinet (task logic)
|                              |
+----------+    +-------------+
|    |
v    v
Kitchen Base Class
(fixture mgmt, object placement,
robot positioning)
|
v
robosuite (backend)
(MuJoCo physics, robot models, controllers)
|
v
MuJoCo 3.3.1 (physics)
(contact dynamics, rendering, sensors)


---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `MuJoCo version must be 3.3.1` | `pip install mujoco==3.3.1` |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` |
| Rendering crashes on Mac | Use `mjpython` instead of `python` |
| `GLFW error` on headless server | `export MUJOCO_GL=egl` or `osmesa` |
| Out of GPU memory | Reduce `batch_size` in `configs/diffusion_policy.yaml` |
| Kitchen assets not found | `python -m robocasa.scripts.download_kitchen_assets` |
| `FileNotFoundError: Augmented data not found` | Run `05b_augment_handle_data.py` first |
| `FileNotFoundError: Dataset not found` | Run `04_download_dataset.py` first |
| Training loss is NaN | Lower learning rate (`--lr 5e-5`) |
| Very slow training | Use a GPU; reduce `--epochs` for CPU runs |

---

## References

- [RoboCasa](https://robocasa.ai/) — kitchen simulation benchmark
- [robosuite](https://robosuite.ai/) — robot learning framework
- [Diffusion Policy — Chi et al., 2023](https://diffusion-policy.cs.columbia.edu/)
- [ACT — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [DAgger — Ross et al., 2011](https://arxiv.org/abs/1011.0686)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
