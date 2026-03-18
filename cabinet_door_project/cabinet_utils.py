"""
Shared utilities for the diffusion policy pipeline.

Provides handle-feature extraction (training + inference), policy
construction / checkpoint loading, and an action-chunking inference runner
so that 09_train, 07_evaluate, and 08_visualize share a single source of
truth for observation ordering and policy interface.
"""

import collections
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Canonical observation key order -- must match parquet column concat order
# used by the training dataset (CabinetAugmentedDataset).
# ---------------------------------------------------------------------------

ROBOT_OBS_KEYS = [
    "robot0_base_pos",         # (3,)
    "robot0_base_quat",        # (4,)
    "robot0_base_to_eef_pos",  # (3,)
    "robot0_base_to_eef_quat", # (4,)
    "robot0_gripper_qpos",     # (2,)
]
ROBOT_STATE_DIM = 16

HANDLE_OBS_COLS = [
    "observation.handle_pos",          # (3,)
    "observation.handle_to_eef_pos",   # (3,)
    "observation.door_openness",       # (1,)
    "observation.handle_xaxis",        # (3,)
    "observation.hinge_direction",     # (1,)
]
HANDLE_FEATURE_DIM = 11

FULL_OBS_DIM = ROBOT_STATE_DIM + HANDLE_FEATURE_DIM  # 27


# ===================================================================
# MuJoCo helper functions (shared with 05b_augment_handle_data.py)
# ===================================================================

def find_fixture_handle_bodies(model, fixture_name):
    """Find MuJoCo body names for the target fixture's door handles."""
    handle_bodies = []
    for i in range(model.nbody):
        name = model.body(i).name
        if fixture_name in name and "handle" in name:
            handle_bodies.append(name)
    return handle_bodies


def find_fixture_door_joints(model, fixture_name):
    """Find door hinge joint names for the target fixture."""
    joints = []
    for i in range(model.njnt):
        jname = model.joint(i).name
        if fixture_name in jname and "door" in jname:
            joints.append((jname, i))
    return joints


def compute_door_openness(model, data, door_joints):
    """Compute average normalized door openness (0=closed, 1=fully open)."""
    if not door_joints:
        return 0.0
    openness_vals = []
    for _jname, jidx in door_joints:
        addr = model.joint(jidx).qposadr[0]
        qpos = data.qpos[addr]
        jrange = model.jnt_range[jidx]
        jmin, jmax = float(jrange[0]), float(jrange[1])
        if jmax - jmin > 1e-8:
            if abs(jmin) < abs(jmax):
                norm = abs(qpos - jmin) / (jmax - jmin)
            else:
                norm = abs(qpos - jmax) / (jmax - jmin)
        else:
            norm = 0.0
        openness_vals.append(np.clip(norm, 0.0, 1.0))
    return float(np.mean(openness_vals))


def build_handle_to_joint_map(handle_bodies, door_joints):
    """Map each handle body to its associated door joint(s)."""
    if len(handle_bodies) == 1 or len(door_joints) == 1:
        return {hb: door_joints for hb in handle_bodies}
    result = {}
    for hb in handle_bodies:
        hb_lower = hb.lower()
        if "left" in hb_lower:
            matched = [(jn, ji) for jn, ji in door_joints if "left" in jn.lower()]
        elif "right" in hb_lower:
            matched = [(jn, ji) for jn, ji in door_joints if "right" in jn.lower()]
        else:
            matched = []
        result[hb] = matched if matched else door_joints
    return result


def get_hinge_direction(handle_body, handle_to_joint_map, model):
    """Return hinge direction (+1 right-opening, -1 left-opening)."""
    joints = handle_to_joint_map.get(handle_body, [])
    if not joints:
        return 0.0
    _, jidx = joints[0]
    jrange = model.jnt_range[jidx]
    jmin, jmax = float(jrange[0]), float(jrange[1])
    return 1.0 if abs(jmin) < abs(jmax) else -1.0


# ===================================================================
# HandleFeatureExtractor  -- live sim feature extraction
# ===================================================================

OPEN_THRESHOLD = 0.90


class HandleFeatureExtractor:
    """
    Extracts handle-related features from a *live* RoboCasa environment.

    Initialise after ``env.reset()`` (which determines the fixture layout).
    Then call ``get_features(env)`` every timestep to obtain the 11-dim
    feature vector in the same order used by the augmented parquet dataset.
    """

    def __init__(self, env):
        ep_meta = env.get_ep_meta()
        fixture_refs = ep_meta.get("fixture_refs", {})
        self.fixture_name = fixture_refs.get("fxtr")
        mj_model = env.sim.model

        if self.fixture_name:
            self.handle_bodies = find_fixture_handle_bodies(mj_model, self.fixture_name)
            self.door_joints = find_fixture_door_joints(mj_model, self.fixture_name)
            self.h2j_map = build_handle_to_joint_map(self.handle_bodies, self.door_joints)
        else:
            self.handle_bodies = []
            self.door_joints = []
            self.h2j_map = {}

    def get_features(self, env) -> np.ndarray:
        """Return (11,) float32 feature vector."""
        mj_model = env.sim.model
        mj_data = env.sim.data

        if not self.handle_bodies:
            return np.zeros(HANDLE_FEATURE_DIM, dtype=np.float32)

        eef_pos = mj_data.body("gripper0_right_eef").xpos.copy()

        per_door = {
            hb: compute_door_openness(mj_model, mj_data, self.h2j_map[hb])
            for hb in self.handle_bodies
        }
        active = [hb for hb in self.handle_bodies if per_door[hb] < OPEN_THRESHOLD]
        candidates = active if active else self.handle_bodies
        dists = [np.linalg.norm(mj_data.body(hb).xpos - eef_pos) for hb in candidates]
        target = candidates[int(np.argmin(dists))]

        handle_pos = mj_data.body(target).xpos.copy().astype(np.float32)
        handle_to_eef = (handle_pos - eef_pos).astype(np.float32)
        openness = np.array([per_door[target]], dtype=np.float32)
        xmat = mj_data.body(target).xmat.reshape(3, 3)
        handle_xaxis = xmat[:, 0].copy().astype(np.float32)
        hinge_dir = np.array(
            [get_hinge_direction(target, self.h2j_map, mj_model)],
            dtype=np.float32,
        )

        return np.concatenate([handle_pos, handle_to_eef, openness, handle_xaxis, hinge_dir])


# ===================================================================
# State extraction helpers
# ===================================================================

def extract_robot_state(obs: dict) -> np.ndarray:
    """
    Build the 16-dim robot state vector from an env observation dict,
    in the canonical order matching ``observation.state`` in the parquet.
    """
    parts = []
    for key in ROBOT_OBS_KEYS:
        val = obs.get(key)
        if val is None:
            raise KeyError(f"Observation missing expected key: {key}")
        parts.append(np.asarray(val, dtype=np.float32).flatten())
    return np.concatenate(parts)


def extract_full_obs(obs: dict, handle_extractor: HandleFeatureExtractor, env) -> np.ndarray:
    """Return the full 27-dim observation vector (robot state + handle features)."""
    robot = extract_robot_state(obs)
    handle = handle_extractor.get_features(env)
    return np.concatenate([robot, handle]).astype(np.float32)


# ===================================================================
# Policy construction & checkpoint loading
# ===================================================================

def _add_diffusion_policy_to_path():
    """Ensure the local diffusion_policy package is importable."""
    dp_root = os.path.join(os.path.dirname(__file__), "diffusion_policy")
    if dp_root not in sys.path:
        sys.path.insert(0, dp_root)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_diffusion_policy(cfg: dict):
    """
    Construct a DiffusionUnetLowdimPolicy + ConditionalUnet1D from a flat
    YAML config dict (no Hydra).
    """
    _add_diffusion_policy_to_path()
    from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    obs_dim = cfg["obs_dim"]
    action_dim = cfg["action_dim"]
    horizon = cfg["horizon"]
    n_obs_steps = cfg["n_obs_steps"]
    n_action_steps = cfg["n_action_steps"]
    pcfg = cfg["policy"]
    obs_as_global_cond = pcfg.get("obs_as_global_cond", True)
    obs_as_local_cond = pcfg.get("obs_as_local_cond", False)
    pred_action_steps_only = pcfg.get("pred_action_steps_only", False)

    if obs_as_global_cond:
        unet_global_cond_dim = obs_dim * n_obs_steps
    else:
        unet_global_cond_dim = None

    if obs_as_local_cond:
        unet_local_cond_dim = obs_dim
    else:
        unet_local_cond_dim = None

    if obs_as_global_cond or obs_as_local_cond:
        unet_input_dim = action_dim
    else:
        unet_input_dim = action_dim + obs_dim

    model = ConditionalUnet1D(
        input_dim=unet_input_dim,
        local_cond_dim=unet_local_cond_dim,
        global_cond_dim=unet_global_cond_dim,
        diffusion_step_embed_dim=pcfg.get("diffusion_step_embed_dim", 256),
        down_dims=pcfg.get("down_dims", [256, 512, 1024]),
        kernel_size=pcfg.get("kernel_size", 3),
        n_groups=pcfg.get("n_groups", 8),
        cond_predict_scale=pcfg.get("cond_predict_scale", False),
    )

    ncfg = cfg["noise_scheduler"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=ncfg.get("num_train_timesteps", 100),
        beta_start=ncfg.get("beta_start", 0.0001),
        beta_end=ncfg.get("beta_end", 0.02),
        beta_schedule=ncfg.get("beta_schedule", "squaredcos_cap_v2"),
        variance_type=ncfg.get("variance_type", "fixed_small"),
        clip_sample=ncfg.get("clip_sample", True),
        prediction_type=ncfg.get("prediction_type", "epsilon"),
    )

    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=ncfg.get("num_inference_steps", None),
        obs_as_local_cond=obs_as_local_cond,
        obs_as_global_cond=obs_as_global_cond,
        pred_action_steps_only=pred_action_steps_only,
    )
    return policy


def load_diffusion_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple["DiffusionUnetLowdimPolicy", dict]:
    """
    Load a checkpoint saved by 09_train_lowdim_unet.py and return
    ``(policy_on_device, cfg_dict)``.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    policy = build_diffusion_policy(cfg)

    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    print(f"Loaded diffusion policy from: {checkpoint_path}")
    print(f"  Epoch {ckpt.get('epoch', '?')}, "
          f"train_loss={ckpt.get('train_loss', float('nan')):.6f}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")
    print(f"  obs_dim={cfg['obs_dim']}, action_dim={cfg['action_dim']}, "
          f"horizon={cfg['horizon']}, n_obs_steps={cfg['n_obs_steps']}, "
          f"n_action_steps={cfg['n_action_steps']}")
    return policy, cfg


# ===================================================================
# DiffusionPolicyRunner  -- action-chunking inference wrapper
# ===================================================================

class DiffusionPolicyRunner:
    """
    Wraps a ``DiffusionUnetLowdimPolicy`` for action-chunking rollouts.

    Maintains an observation FIFO (last *n_obs_steps* observations) and an
    action queue.  Call ``get_action()`` every environment step; inference
    is triggered only when the action queue is empty.
    """

    def __init__(self, policy, cfg: dict, device: torch.device):
        self.policy = policy
        self.device = device
        self.n_obs_steps: int = cfg["n_obs_steps"]
        self.n_action_steps: int = cfg["n_action_steps"]
        self.obs_dim: int = cfg["obs_dim"]
        self.action_dim: int = cfg["action_dim"]

        self.obs_buffer: collections.deque = collections.deque(maxlen=self.n_obs_steps)
        self.action_queue: collections.deque = collections.deque()

    def reset(self):
        """Call at the start of each episode."""
        self.obs_buffer.clear()
        self.action_queue.clear()

    def get_action(self, obs: np.ndarray, handle_extractor: HandleFeatureExtractor,
                   env) -> np.ndarray:
        """
        Return a single (action_dim,) action for one environment step.

        Parameters
        ----------
        obs : np.ndarray
            Raw environment observation dict has already been converted to
            the full 27-dim vector via ``extract_full_obs`` by the caller,
            *or* the caller passes the raw dict and we do it here.
        """
        if isinstance(obs, dict):
            obs = extract_full_obs(obs, handle_extractor, env)

        self.obs_buffer.append(obs.copy())

        if len(self.action_queue) == 0:
            obs_seq = self._build_obs_tensor()
            with torch.no_grad():
                obs_dict = {"obs": obs_seq.unsqueeze(0).to(self.device)}
                result = self.policy.predict_action(obs_dict)
            actions = result["action"].cpu().numpy().squeeze(0)  # (n_action_steps, Da)
            for a in actions:
                self.action_queue.append(a)

        return self.action_queue.popleft()

    def _build_obs_tensor(self) -> torch.Tensor:
        """Pad / stack the observation buffer to (n_obs_steps, obs_dim)."""
        buf = list(self.obs_buffer)
        while len(buf) < self.n_obs_steps:
            buf.insert(0, buf[0].copy())
        arr = np.stack(buf[-self.n_obs_steps:], axis=0).astype(np.float32)
        return torch.from_numpy(arr)
