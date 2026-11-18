"""Motions: rotations, making, joining and stretching md.Motion, feet, dict round trip.

Moved from the benchmark notebook (pupper_generation_benchmark.ipynb), cell "Core:
keyframe format, interpolation, rig map, physics gate, metrics", section "rotations /
motions", unchanged. Everything is plain CPU code on top of pupper-mjlab's
motion_design.py (md).
"""
import math

import numpy as np

from mjlab.tasks.tracking.config.pupper import motion_design as md

FPS = 50.0  # Pupper motions
LEGS = md.LEGS
H = md.STAND_HEIGHT
KIN = md.kinematics()
STAND_FEET = {leg: np.array(p, float) for leg, p in md.stand_feet().items()}


def quat_to_rpy(q) -> np.ndarray:
  """(w, x, y, z) -> (roll, pitch, yaw) degrees; inverse of md.rpy_to_quat (z-y-x)."""
  R = md.quat_to_mat(q)
  yaw = math.atan2(R[1, 0], R[0, 0])
  pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
  roll = math.atan2(R[2, 1], R[2, 2])
  return np.degrees([roll, pitch, yaw])


def rpy_series(motion) -> np.ndarray:
  rpy = np.array([quat_to_rpy(q) for q in motion.root_quat])
  rpy[:, 2] = np.degrees(np.unwrap(np.radians(rpy[:, 2])))
  return rpy


def quat_angle_deg(a, b) -> float:
  a, b = np.asarray(a, float), np.asarray(b, float)
  dot = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
  return math.degrees(2 * math.acos(min(1.0, dot)))


def make_motion(root_pos, root_quat, joint_pos) -> md.Motion:
  m = md.Motion(
    fps=FPS,
    root_pos=np.asarray(root_pos, float),
    root_quat=np.asarray(root_quat, float),
    joint_pos=np.asarray(joint_pos, float),
  )
  m.report = md.check_motion(m)
  return m


def concat_motions(a: md.Motion, b: md.Motion) -> md.Motion:
  """a then b; b's first frame is dropped (it repeats a's last one)."""
  return make_motion(
    np.concatenate([a.root_pos, b.root_pos[1:]]),
    np.concatenate([a.root_quat, b.root_quat[1:]]),
    np.concatenate([a.joint_pos, b.joint_pos[1:]]),
  )


def stretch_motion(motion: md.Motion, factor: float) -> md.Motion:
  """Play the motion `factor` times slower (same poses, more frames)."""
  n = len(motion.root_pos)
  m = int(round((n - 1) * factor)) + 1
  f = np.linspace(0.0, n - 1, m)
  i0 = np.floor(f).astype(int)
  i1 = np.minimum(i0 + 1, n - 1)
  a = (f - i0)[:, None]
  q0, q1 = motion.root_quat[i0], motion.root_quat[i1].copy()
  q1[(q0 * q1).sum(1) < 0] *= -1
  q = (1 - a) * q0 + a * q1
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  return make_motion(
    (1 - a) * motion.root_pos[i0] + a * motion.root_pos[i1],
    q,
    (1 - a) * motion.joint_pos[i0] + a * motion.joint_pos[i1],
  )


def feet_series(motion) -> np.ndarray:
  """World foot centres (T, 4, 3) in LEGS order."""
  out = np.zeros((len(motion.root_pos), 4, 3))
  for i in range(len(out)):
    f = KIN.feet(motion.root_pos[i], motion.root_quat[i], motion.joint_pos[i])
    out[i] = [f[leg] for leg in LEGS]
  return out


def motion_to_dict(m: md.Motion) -> dict:
  return {"root_pos": m.root_pos, "root_quat": m.root_quat, "joint_pos": m.joint_pos}


def motion_from_dict(d: dict) -> md.Motion:
  return make_motion(d["root_pos"], d["root_quat"], d["joint_pos"])
