"""
Step 8: Visualize a Policy Rollout
=====================================
Loads a trained diffusion policy checkpoint from 09_train_lowdim_unet.py
and runs it live in the OpenCabinet environment so you can watch the robot.

Two rendering modes:
  On-screen  (default)  -- interactive MuJoCo viewer window, real-time
  Off-screen (--offscreen) -- renders to a video file, works without a display

Usage:
    python 08_visualize_policy_rollout.py \\
        --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt

    python 08_visualize_policy_rollout.py --checkpoint ... --offscreen

    python 08_visualize_policy_rollout.py --checkpoint ... --num_episodes 3

    # Mac users must use mjpython for the on-screen window
    mjpython 08_visualize_policy_rollout.py --checkpoint ...
"""

import os
import sys

# ── Rendering mode detection ────────────────────────────────────────────────
_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
    if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
        _env = dict(os.environ)
        _changed = False
        if _env.get("WAYLAND_DISPLAY"):
            if not _env.get("DISPLAY", "").startswith(":"):
                _env["DISPLAY"] = ":0"
                _changed = True
            if _env.get("GALLIUM_DRIVER") != "llvmpipe":
                _env["GALLIUM_DRIVER"] = "llvmpipe"
                _changed = True
            if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
                _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
                _changed = True
        if _changed:
            _env["__TELEOP_DISPLAY_OK"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, _env)
        else:
            os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import argparse
import time

import numpy as np
import robocasa  # noqa: F401 -- registers OpenCabinet environment
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

from cabinet_utils import (
    HandleFeatureExtractor,
    DiffusionPolicyRunner,
    extract_full_obs,
    load_diffusion_checkpoint,
    check_any_door_open,
)


# ── On-screen rollout ────────────────────────────────────────────────────────

def run_onscreen(policy, cfg, device, args):
    """
    Run the policy with an interactive MuJoCo viewer window.
    """
    env = robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    env = VisualizationWrapper(env)

    successes = 0
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")
        print(f"  Running for up to {args.max_steps} steps...")
        print(f"  (Watch the viewer window -- use mouse to orbit the camera)\n")

        handle_ext = HandleFeatureExtractor(env)
        runner = DiffusionPolicyRunner(policy, cfg, device)
        runner.reset()

        success = False
        hold_count = 0

        for step in range(args.max_steps):
            full_obs = extract_full_obs(obs, handle_ext, env)
            action = runner.get_action(full_obs, handle_ext, env)

            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            action = action * args.action_scale
            obs, reward, done, info = env.step(action)

            if step % 20 == 0:
                checking = check_any_door_open(handle_ext, env)
                status = "cabinet OPEN" if checking else "in progress"
                act_mag = float(np.abs(action).mean())
                print(
                    f"  step {step:4d}  reward={reward:+.3f}  "
                    f"action_mag={act_mag:.3f}  [{status}]"
                )

            if check_any_door_open(handle_ext, env):
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

            time.sleep(1.0 / args.max_fr)

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"\n  Result: {result}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Off-screen rollout with video ────────────────────────────────────────────

def run_offscreen(policy, cfg, device, args):
    """
    Run the policy headlessly and save an annotated video.
    """
    import imageio
    from robocasa.utils.env_utils import create_env

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768

    successes = 0
    all_frames = []

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=args.seed + ep,
            camera_widths=cam_w,
            camera_heights=cam_h,
        )
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")

        handle_ext = HandleFeatureExtractor(env)
        runner = DiffusionPolicyRunner(policy, cfg, device)
        runner.reset()

        success = False
        hold_count = 0
        ep_frames = []

        for step in range(args.max_steps):
            full_obs = extract_full_obs(obs, handle_ext, env)
            action = runner.get_action(full_obs, handle_ext, env)

            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            action = action * args.action_scale
            obs, reward, done, info = env.step(action)

            frame = env.sim.render(
                height=cam_h, width=cam_w, camera_name="robot0_agentview_center"
            )[::-1]
            ep_frames.append(frame)

            if step % 20 == 0:
                checking = check_any_door_open(handle_ext, env)
                status = "cabinet OPEN" if checking else "in progress"
                print(f"  step {step:4d}  reward={reward:+.3f}  [{status}]")

            if check_any_door_open(handle_ext, env):
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"  Result: {result}  ({len(ep_frames)} frames)")
        if success:
            successes += 1

        all_frames.extend(ep_frames)
        env.close()

    print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
    with imageio.get_writer(args.video_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video_path}")

    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained diffusion policy rollout in OpenCabinet"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt",
        help="Path to diffusion policy checkpoint (.pt) saved by 09_train_lowdim_unet.py",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1, help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps", type=int, default=300, help="Maximum steps per episode",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Render to video file instead of opening a viewer window",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="tmp/policy_rollout.mp4",
        help="Output video path (used with --offscreen)",
    )
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument(
        "--max_fr",
        type=int,
        default=20,
        help="On-screen playback rate cap (frames/second); lower = slower",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--action_scale", type=float, default=1.0,
        help="Multiply actions by this factor before env.step (e.g. 0.5 to halve)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Rollout Visualizer (Diffusion UNet)")
    print("=" * 60)
    print()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required.  Run: pip install torch")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Train a policy first with:  python 09_train_lowdim_unet.py")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    policy, cfg = load_diffusion_checkpoint(args.checkpoint, device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print()

    mode = "off-screen (video)" if args.offscreen else "on-screen (viewer window)"
    print(f"Mode:     {mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps/ep: {args.max_steps}")
    if args.offscreen:
        print(f"Output:   {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(policy, cfg, device, args)
    else:
        print("Opening viewer window...")
        print("  Tip: orbit the camera with the mouse to see the gripper.\n")
        run_onscreen(policy, cfg, device, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
