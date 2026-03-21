# Cabinet Door Opening Robot - CS 188 Starter Project

### Disclaimer

This project was designed for CS 188 - Intro to Robotics as a template starter project. If you have any issues with the codebase, please email me at holdengs @ cs.ucla.edu!

## Overview

In this project you will build a robot that learns to open kitchen cabinet doors
using **RoboCasa365**, a large-scale simulation benchmark for everyday robot
tasks. You will progress from understanding the simulation environment, to
collecting demonstrations, to training a neural-network policy that controls the
robot autonomously.

### What you will learn

1. How robotic manipulation environments are structured (MuJoCo + robosuite + RoboCasa)
2. How the `OpenCabinet` task works -- sensors, actions, success criteria
3. How to collect and use demonstration datasets (human + MimicGen)
4. How to train a behavior-cloning policy from demonstrations
5. How to evaluate your trained policy in simulation

### The robot

We use the **PandaOmron** mobile manipulator -- a Franka Panda 7-DOF arm
mounted on an Omron wheeled base with a torso lift joint. This is the default
and best-supported robot in RoboCasa.

---

## Installation

Run the install script (works on **macOS** and **WSL/Linux**):

```bash
./install.sh
```

This will:
- Create a Python virtual environment (`.venv`)
- Clone and install robosuite and robocasa
- Install all Python dependencies (PyTorch, numpy, matplotlib, etc.)
- Download RoboCasa kitchen assets (~10 GB)

After installation, activate the environment:

```bash
source .venv/bin/activate
```

Then verify everything works:

```bash
cd cabinet_door_project
python 00_verify_installation.py
```

> **macOS note:** Scripts that open a rendering window (03, 05, 08) require
> `mjpython` instead of `python`. The install script will remind you of this.

---

## Project Structure

```
cabinet_door_project/
  00_verify_installation.py        # Check that everything is installed correctly
  01_explore_environment.py        # Create the OpenCabinet env, inspect observations/actions
  02_random_rollouts.py            # Run random actions, save video, understand the task
  03_teleop_collect_demos.py       # Teleoperate the robot to collect your own demonstrations
  04_download_dataset.py           # Download the pre-collected OpenCabinet dataset
  05_playback_demonstrations.py    # Play back demonstrations to see expert behavior
  05b_augment_handle_data.py       # Augment dataset with handle position & door features
  06_train_policy.py               # Train a simple MLP behavior-cloning policy (baseline)
  07_evaluate_policy.py            # Evaluate your trained policy in simulation
  08_visualize_policy_rollout.py   # Visualize a rollout of your policy in RoboCasa
  09_train_lowdim_unet.py          # Train a diffusion policy (ConditionalUnet1D)
  cabinet_utils.py                 # Shared utilities (handle extraction, policy construction)
  configs/
    diffusion_policy.yaml          # Diffusion policy hyperparameters
  figures/                         # Generated plots and diagrams
  plot_generators/                 # Scripts to regenerate figures
  notebook.ipynb                   # Interactive Jupyter notebook companion
install.sh                         # Installation script (macOS + WSL/Linux)
README.md                          # This file
```

---

## Step-by-Step Guide

### Step 0: Verify Installation

```bash
python 00_verify_installation.py
```

This checks that MuJoCo, robosuite, RoboCasa, and all dependencies are
correctly installed and that the `OpenCabinet` environment can be created.

### Step 1: Explore the Environment

```bash
python 01_explore_environment.py
```

This script creates the `OpenCabinet` environment and prints detailed
information about:
- **Observation space**: what the robot sees (camera images, joint positions,
  gripper state, base pose)
- **Action space**: what the robot can do (arm movement, gripper open/close,
  base motion, control mode)
- **Task description**: the natural language instruction for the episode
- **Success criteria**: how the environment determines task completion

### Step 2: Random Rollouts

```bash
python 02_random_rollouts.py
```

Runs the robot with random actions to see what happens (spoiler: nothing
useful, but it helps you understand the action space). Saves a video to
`/tmp/cabinet_random_rollouts.mp4`.

### Step 3: Teleoperate and Collect Demonstrations

```bash
# Mac users: use mjpython instead of python
python 03_teleop_collect_demos.py
```

Control the robot yourself using the keyboard to open cabinet doors. This
gives you intuition for the task difficulty and generates demonstration data.

**Keyboard controls:**
| Key | Action |
|-----|--------|
| Ctrl+q | Reset simulation |
| spacebar | Toggle gripper (open/close) |
| up-right-down-left | Move horizontally in x-y plane |
| .-; | Move vertically |
| o-p | Rotate (yaw) |
| y-h | Rotate (pitch) |
| e-r | Rotate (roll) |
| b | Toggle arm/base mode (if applicable) |
| s | Switch active arm (if multi-armed robot) |
| = | Switch active robot (if multi-robot environment) |              

### Step 4: Download Pre-collected Dataset

```bash
python 04_download_dataset.py
```

Downloads the official OpenCabinet demonstration dataset from the RoboCasa
servers. This includes both human demonstrations and MimicGen-expanded data
across diverse kitchen scenes.

### Step 5: Play Back Demonstrations

```bash
python 05_playback_demonstrations.py
```

Visualize the downloaded demonstrations to see how an expert opens cabinet
doors. This is the data your policy will learn from.

### Step 5b: Augment the Dataset with Handle Features

```bash
python 05b_augment_handle_data.py
```

Replays the saved MuJoCo states from the demonstration dataset to extract
handle position, door openness, and other features the robot needs. It writes
augmented parquet files alongside the originals.

Output: an `augmented/` directory inside the dataset folder containing one
parquet file per episode, each with the original columns plus 5 new
observation columns (11 extra dims total).

You only need to run this once. If you already have the augmented parquet
files, skip this step.

### Step 6: Train a Baseline Policy

```bash
python 06_train_policy.py
```

Trains a simple MLP behavior-cloning policy on low-dimensional state-action
pairs from the demonstration data. This is meant to illustrate the
data-loading → training → checkpoint pipeline, not to produce a policy that
can reliably solve the task.

---

## Diffusion Policy Pipeline

The main training approach uses a diffusion policy (ConditionalUnet1D) trained
on the augmented low-dimensional dataset. All commands are run from the
`cabinet_door_project/` directory with the virtual environment active.

```
Step 5b   05b_augment_handle_data.py       Add handle features to parquet (run once)
   |
Step 9    09_train_lowdim_unet.py          Train the diffusion policy
   |
Step 7    07_evaluate_policy.py            Evaluate success rate
   |
Step 8    08_visualize_policy_rollout.py   Watch the robot
```

### Train the Diffusion Policy

```bash
# Default (uses configs/diffusion_policy.yaml)
python 09_train_lowdim_unet.py

# Override hyperparameters from the command line
python 09_train_lowdim_unet.py --epochs 100 --batch_size 64
python 09_train_lowdim_unet.py --lr 3e-4
python 09_train_lowdim_unet.py --checkpoint_dir ./my_checkpoints
```

#### What gets saved

| File | Description |
|------|-------------|
| `best_diffusion_policy.pt` | Best checkpoint by validation loss |
| `final_diffusion_policy.pt` | Checkpoint at the end of training |
| `checkpoint_epoch_XXXXX.pt` | Periodic checkpoints (every N epochs) |

Each checkpoint contains the full policy weights, normalizer state, optimizer
state, and the config dict so it is fully self-contained.

#### Key config parameters (`configs/diffusion_policy.yaml`)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `horizon` | 16 | Trajectory length the UNet denoises (timesteps) |
| `n_obs_steps` | 2 | Number of past observations used as context |
| `n_action_steps` | 8 | Number of actions executed before re-planning |
| `training.num_epochs` | 100 | Total training epochs |
| `training.batch_size` | 64 | Batch size (reduce if you run out of memory) |
| `training.learning_rate` | 1e-4 | Learning rate (cosine decay) |
| `training.use_ema` | true | Exponential Moving Average of weights |
| `noise_scheduler.num_train_timesteps` | 100 | DDPM diffusion steps |

### Evaluate the Policy

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
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--num_rollouts` | 20 | Number of episodes to run |
| `--max_steps` | 500 | Max timesteps per episode |
| `--split` | `pretrain` | `pretrain` or `target` (held-out kitchens) |
| `--video_path` | None | If set, saves an MP4 of the evaluation |
| `--seed` | 0 | Random seed for environment layout |

### Visualize the Policy

```bash
# On-screen (interactive viewer window)
# macOS: use mjpython instead of python
mjpython 08_visualize_policy_rollout.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt

# Off-screen (headless, saves video)
python 08_visualize_policy_rollout.py \
    --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt \
    --offscreen --video_path tmp/my_rollout.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `best_diffusion_policy.pt` | Path to `.pt` checkpoint |
| `--num_episodes` | 1 | Episodes to run |
| `--max_steps` | 300 | Max timesteps per episode |
| `--offscreen` | off | Render to video instead of viewer window |
| `--video_path` | `tmp/policy_rollout.mp4` | Output video path |
| `--fps` | 20 | Video frames per second |
| `--seed` | 42 | Random seed |

---

## Key Concepts

### The OpenCabinet Task

- **Goal**: Open a kitchen cabinet door
- **Fixture**: `HingeCabinet` (a cabinet with hinged doors)
- **Initial state**: Cabinet door is closed; robot is positioned nearby
- **Success**: `fixture.is_open(env)` returns `True`
- **Horizon**: 500 timesteps at 20 Hz control frequency (25 seconds)
- **Scene variety**: 2,500+ kitchen layouts/styles for generalization

### Observation Space (PandaOmron)

| Key | Shape | Description |
|-----|-------|-------------|
| `robot0_agentview_left_image` | (256, 256, 3) | Left shoulder camera |
| `robot0_agentview_right_image` | (256, 256, 3) | Right shoulder camera |
| `robot0_eye_in_hand_image` | (256, 256, 3) | Wrist-mounted camera |
| `robot0_gripper_qpos` | (2,) | Gripper finger positions |
| `robot0_base_pos` | (3,) | Base position (x, y, z) |
| `robot0_base_quat` | (4,) | Base orientation quaternion |
| `robot0_base_to_eef_pos` | (3,) | End-effector pos relative to base |
| `robot0_base_to_eef_quat` | (4,) | End-effector orientation relative to base |

### Action Space (PandaOmron)

| Key | Dim | Description |
|-----|-----|-------------|
| `end_effector_position` | 3 | Delta (dx, dy, dz) for the end-effector |
| `end_effector_rotation` | 3 | Delta rotation (axis-angle) |
| `gripper_close` | 1 | 0 = open, 1 = close |
| `base_motion` | 4 | (forward, side, yaw, torso) |
| `control_mode` | 1 | 0 = arm control, 1 = base control |

### Dataset Format (LeRobot)

Datasets are stored in LeRobot format:
```
dataset/
  meta/           # Episode metadata (task descriptions, camera info)
  videos/         # MP4 videos from each camera
  data/           # Parquet files with actions, states, rewards
  extras/         # Per-episode metadata (MuJoCo states, model XML)
```

---

## Architecture Diagram

```
                    RoboCasa Stack
                    ==============

  +-------------------+     +-------------------+
  |   Kitchen Scene   |     |   OpenCabinet     |
  |  (2500+ layouts)  |     |   (Task Logic)    |
  +--------+----------+     +--------+----------+
           |                         |
           v                         v
  +------------------------------------------------+
  |              Kitchen Base Class                 |
  |  - Fixture management (cabinets, fridges, etc)  |
  |  - Object placement (bowls, cups, etc)          |
  |  - Robot positioning                            |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              robosuite (Backend)                |
  |  - MuJoCo physics simulation                   |
  |  - Robot models (PandaOmron, GR1, Spot, ...)   |
  |  - Controller framework                        |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              MuJoCo 3.3.1 (Physics)            |
  |  - Contact dynamics, rendering, sensors        |
  +------------------------------------------------+
```

---

## Research Directions

The MLP baseline in `06_train_policy.py` is intentionally simple — it
demonstrates the pipeline but will basically always fail. Here are three
fun directions to improve the model:

### Minimal Diffusion Policy

Replace the direct-regression MLP with a diffusion-based action generator.
The core loop is to corrupt ground-truth actions with Gaussian noise,
train the network to predict that noise conditioned on the current state, and
at inference iteratively denoise from pure noise to produce an action. This
properly handles multi-modal demonstrations (e.g., approaching the handle from
the left vs. right) that MSE loss averages into useless mean actions.
See [Chi et al., 2023](https://diffusion-policy.cs.columbia.edu/) for the
full approach — a minimal version can be built in ~100 lines on top of the
existing MLP backbone.

### DAgger (Online Correction)

Script 03 already provides keyboard teleoperation. I have it set up with a DAgger mode that may or may not be kinda buggy. Use it to close the loop:
train a policy, roll it out, then have a human take over and correct the robot
whenever it fails. Aggregate these corrections into the training set and
retrain. This directly attacks distribution shift — the fundamental reason
offline BC degrades at test time — by collecting data in the states the policy
actually visits. Even one or two rounds of DAgger can dramatically improve
robustness. See [Ross et al., 2011](https://arxiv.org/abs/1011.0686).

### Action Chunking

Instead of predicting one action per timestep, predict the next *K* actions at
once and execute them open-loop before re-planning. This is the key idea behind
ACT ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) and directly fixes
the jerky, temporally incoherent behavior of single-step BC. Fair warning, though, this will probably require a more sophisticated model (Transformer, Diffusion or other) to provide real benefits. Implementation is
straightforward: widen the output head to `K * action_dim`, train with the same
MSE loss over the full chunk, and add a small FIFO buffer at inference. Try
sweeping K = 4, 8, 16 and compare smoothness and success rate.

### Other Ideas
- Gaussian Mixture Model for output logits. Can ameliorate the MSE multimodality issue.
- Vision Transformer. Will need a beefier computer to see benefits but definitely can improve policy at scale.
- Hooking in an existing VLM and experimenting with zero-shot inference.

---

## Troubleshooting

I'll continually update this section as students find bugs in the system. Please, let me know if you encounter issues!

| Problem | Solution |
|---------|----------|
| `MuJoCo version must be 3.3.1` | `pip install mujoco==3.3.1` |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` |
| Rendering crashes on Mac | Use `mjpython` instead of `python` |
| `GLFW error` on headless server | Set `export MUJOCO_GL=egl` or `osmesa` |
| Out of GPU memory during training | Reduce batch size in `configs/diffusion_policy.yaml` |
| Kitchen assets not found | Run `python -m robocasa.scripts.download_kitchen_assets` |
| `FileNotFoundError: Augmented data not found` | Run `python 05b_augment_handle_data.py` first |
| `FileNotFoundError: Dataset not found` | Run `python 04_download_dataset.py` first |
| Training loss is NaN | Lower learning rate (try `--lr 5e-5`) |
| Very slow training on CPU | Expected. Use a GPU if available, or reduce `--epochs` |

---

## File Reference

| File | Purpose |
|------|---------|
| `cabinet_utils.py` | Shared utilities (handle extraction, policy construction, inference runner) |
| `configs/diffusion_policy.yaml` | All hyperparameters for diffusion policy training |
| `05b_augment_handle_data.py` | One-time data augmentation (handle + door features) |
| `09_train_lowdim_unet.py` | Diffusion policy training script |
| `07_evaluate_policy.py` | Quantitative evaluation (success rate) |
| `08_visualize_policy_rollout.py` | Visual debugging (viewer or video) |
| `plot_generators/` | Scripts to regenerate figures (training curves, architecture, etc.) |
| `figures/` | Generated plots and diagrams |

---

## References

- [RoboCasa Paper & Website](https://robocasa.ai/)
- [RoboCasa GitHub](https://github.com/robocasa/robocasa)
- [robosuite Documentation](https://robosuite.ai/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
