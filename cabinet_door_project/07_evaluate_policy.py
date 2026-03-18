"""
Step 7: Evaluate a Trained Policy
===================================
Runs a trained diffusion policy in the OpenCabinet environment and reports
success rate across multiple episodes and kitchen scenes.

The policy is loaded from a checkpoint saved by 09_train_lowdim_unet.py.
At each timestep the 27-dim observation (robot state + handle features) is
extracted from the live simulation and fed through the action-chunking
inference runner.

Usage:
    python 07_evaluate_policy.py --checkpoint tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt

    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --num_rollouts 50

    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --split target

    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --video_path tmp/eval_videos.mp4
"""

import argparse
import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env

from cabinet_utils import (
    HandleFeatureExtractor,
    DiffusionPolicyRunner,
    extract_full_obs,
    load_diffusion_checkpoint,
)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_evaluation(policy, cfg, device, num_rollouts, max_steps, split, video_path, seed, action_scale=1.0):
    """Run evaluation rollouts and collect statistics."""
    import imageio

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {"successes": [], "episode_lengths": [], "rewards": []}

    for ep in range(num_rollouts):
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        handle_ext = HandleFeatureExtractor(env)
        runner = DiffusionPolicyRunner(policy, cfg, device)
        runner.reset()

        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            full_obs = extract_full_obs(obs, handle_ext, env)
            action = runner.get_action(full_obs, handle_ext, env)

            env_action_dim = env.action_dim
            if len(action) < env_action_dim:
                action = np.pad(action, (0, env_action_dim - len(action)))
            elif len(action) > env_action_dim:
                action = action[:env_action_dim]

            action = action * action_scale
            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            if env._check_success():
                success = True
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step + 1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to diffusion policy checkpoint (.pt file from 09_train)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument(
        "--video_path", type=str, default=None, help="Path to save evaluation video"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--action_scale", type=float, default=1.0,
        help="Multiply actions by this factor before env.step (e.g. 0.5 to halve)",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Policy Evaluation (Diffusion UNet)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    policy, cfg = load_diffusion_checkpoint(args.checkpoint, device)

    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")

    results = run_evaluation(
        policy=policy,
        cfg=cfg,
        device=device,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
        action_scale=args.action_scale,
    )

    print_section("Evaluation Results")

    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")

    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")

    print_section("Performance Context")
    print(
        "Expected success rates from the RoboCasa benchmark:\n"
        "\n"
        "  Method            | Pretrain | Target\n"
        "  ------------------|----------|-------\n"
        "  Random actions    |    ~0%   |   ~0%\n"
        "  Diffusion Policy  |  ~30-60% | ~20-50%\n"
        "  pi-0              |  ~40-70% | ~30-60%\n"
        "  GR00T N1.5        |  ~35-65% | ~25-55%\n"
    )


if __name__ == "__main__":
    main()
