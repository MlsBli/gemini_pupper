"""An extra reward term for the tracking task (v2): paw height.

New after Phase 4. v1, trained with the course config, skips small paw lifts: of 14
test motions that lift a paw, it did a median 14% of the lift (plain replay: 73%).
The course's body-position reward averages 9 bodies with a 10 cm tolerance, so a
5 cm paw left on the ground costs ~3% of one term. This term looks only at what was
being skipped: the *height* of each paw, compared with the reference's.

The tracked "foot" bodies (leg_*_3) are the lower legs, whose frames sit at the
knees; the paw centre is a fixed offset in that frame (the model's foot site). So
the paw point is computed from each lower leg's pose, for the reference and the
robot alike. Height only: the reference's horizontal position follows the robot's
(the task re-anchors it every step), and sideways drift is the torso terms' job.

  reward = exp(-mean over the 4 paws of (z_reference - z_robot)^2 / std^2)

With std = 0.02 m, one paw left on the ground during a 5 cm lift scores
exp(-(0.05^2 / 4) / 0.02^2) = 0.21 instead of 1; a 1 cm miss on one paw scores 0.94.
"""
from typing import cast

import mujoco
import torch

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import quat_apply


def paw_offsets() -> dict[str, list[float]]:
  """Lower-leg body name -> the paw centre in that body's frame (m), from the model."""
  k = md.kinematics()
  m = k.model
  return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[sid])): m.site_pos[sid].tolist()
          for sid in k.foot_site.values()}


PAW_OFFSETS = paw_offsets()
_CACHE: dict = {}


def paw_points(command: MotionCommand, robot: bool) -> torch.Tensor:
  """(num_envs, 4, 3) paw centres in the world: the robot's, or the reference's (re-anchored
  to the robot like every body the task compares)."""
  key = (id(command), str(command.device))
  if key not in _CACHE:
    names = list(command.cfg.body_names)
    idx = torch.tensor([names.index(b) for b in PAW_OFFSETS], device=command.device)
    off = torch.tensor(list(PAW_OFFSETS.values()), dtype=torch.float32, device=command.device)
    _CACHE[key] = (idx, off)
  idx, off = _CACHE[key]
  if robot:
    pos, quat = command.robot_body_pos_w[:, idx], command.robot_body_quat_w[:, idx]
  else:
    pos, quat = command.body_pos_relative_w[:, idx], command.body_quat_relative_w[:, idx]
  n = pos.shape[0]
  return pos + quat_apply(quat.reshape(-1, 4), off.expand(n, -1, -1).reshape(-1, 3)).reshape(n, -1, 3)


def paw_height_error_exp(env, command_name: str, std: float) -> torch.Tensor:
  """exp(-mean squared paw-height error / std^2), one value per env (1 = every paw at the
  reference's height)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  err = (paw_points(command, robot=False)[..., 2] - paw_points(command, robot=True)[..., 2]) ** 2
  return torch.exp(-err.mean(-1) / std**2)
