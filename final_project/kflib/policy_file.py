"""A robot trick policy.json for a new motion: the general tracker's weights + that motion.

New in Phase 5 (live path). A general tracker's weights are the same for every motion;
only the embedded reference changes (Phase 0, question 4). So a new motion's policy
file is a copy of any exported policy of that tracker (the template) with its
"motion" and "trick" blocks rebuilt from the motion's npz, exactly as pupper-mjlab's
export_trick writes them (export_trick.py:81-101): the reference joint angles,
joint velocities and base orientation per 50 Hz frame, rounded to 6 decimals. On the
course's own files this rebuild matches export_trick's output exactly.

It takes milliseconds and needs no GPU, no env and no checkpoint.
"""
import copy
import json
from pathlib import Path

import numpy as np


def _rows(a) -> list:
  return np.round(np.asarray(a).astype(float), 6).tolist()  # export_trick.py:81-82


def motion_block(motion_npz) -> dict:
  """The "motion" block for a tracker npz (base_link is body 0)."""
  with np.load(motion_npz) as z:
    if str(z["body_names"][0]) != "base_link":
      raise ValueError(f"{motion_npz}: body 0 is {z['body_names'][0]}, expected base_link")
    return dict(joint_pos=_rows(z["joint_pos"]), joint_vel=_rows(z["joint_vel"]),
                anchor_quat_wxyz=_rows(z["body_quat_w"][:, 0]))


def policy_with_motion(template: dict, motion_npz, name: str) -> dict:
  """The template's policy (weights, gains, layout) playing this motion instead."""
  if "layers" not in template or "trick" not in template or "motion" not in template:
    raise ValueError("the template must be an exported trick policy (layers, trick and motion blocks)")
  policy = copy.deepcopy({k: v for k, v in template.items() if k not in ("motion", "trick")})
  policy["motion"] = motion_block(motion_npz)
  frames = len(policy["motion"]["joint_pos"])
  fps = int(template["trick"].get("fps", 50))
  policy["trick"] = dict(name=name, fps=fps, frames=frames, duration_s=frames / fps)
  return policy


def write_policy(template_json, motion_npz, name: str, out_json) -> dict:
  policy = policy_with_motion(json.loads(Path(template_json).read_text()), motion_npz, name)
  Path(out_json).write_text(json.dumps(policy))
  return policy
