"""The keyframe format Gemini writes, its validation, interpolation and IK.

Moved from the benchmark notebook: cell "Core" (sections "the keyframe format" and
"interpolation") and reach_problems from cell "The three pipelines", unchanged.
"""
import dataclasses
import math

import numpy as np

from mjlab.tasks.tracking.config.pupper import motion_design as md

from .motion import FPS, H, KIN, LEGS, STAND_FEET, quat_angle_deg


# What Gemini writes (one JSON object). Everything is relative to the stand, so
# the numbers are small and readable:
#   {"t": 0.0, "pose": "stand"}
#   {"t": 0.6, "body": {"x": 0, "y": 0, "dz": -0.02, "roll": 0, "pitch": 10, "yaw": 0},
#    "feet": {"front_l": [0.0, 0.0, 0.04]}, "ease": "stop", "hold": 0.5}
# body: x, y = body position (m) from the start; dz = height change from standing;
# roll/pitch/yaw in degrees. feet: [dx, dy, dz] (m, world frame) from where that
# foot stands; feet left out stay planted. ease: "stop" comes to rest at the
# keyframe (default), "flow" passes through it. hold: seconds to stay there.
SCHEMA_LIMITS = dict(  # generous sanity bounds; feasibility is the verifier's job
  t_max=10.0, min_gap=0.08, body_xy=0.10, dz=(-0.13, 0.06),
  roll=35.0, pitch=50.0, yaw=60.0, foot_xy=0.10, foot_dz=(0.0, 0.12),
)


@dataclasses.dataclass
class KF:
  t: float
  body: np.ndarray  # x, y, dz, roll, pitch, yaw
  feet: dict  # leg -> np.array([dx, dy, dz])
  ease: str = "stop"
  src: int = -1  # index in Gemini's list (for error messages)

  def md_keyframe(self) -> md.Keyframe:
    x, y, dz, r, p, yw = (float(v) for v in self.body)
    feet = {}
    for leg in LEGS:
      off = self.feet.get(leg, np.zeros(3))
      s = STAND_FEET[leg]
      feet[leg] = (s[0] + off[0], s[1] + off[1], md.FOOT_RADIUS + off[2])
    return md.Keyframe(self.t, (x, y, H + dz), (r, p, yw), feet=feet)

  def channels(self) -> np.ndarray:
    """18 numbers interpolated between keyframes: root xyz, rpy, 4 feet xyz."""
    k = self.md_keyframe()
    return np.concatenate([k.root_pos, k.root_rpy] + [np.array(k.feet[leg]) for leg in LEGS])


def _num(v, what, errs):
  try:
    x = float(v)
  except (TypeError, ValueError):
    errs.append(f"{what} must be a number, got {v!r}")
    return 0.0
  if not math.isfinite(x):
    errs.append(f"{what} must be finite")
    return 0.0
  return x


def parse_keyframes(items) -> tuple[list[KF], list[str]]:
  """Gemini's keyframe list -> KFs (holds expanded), plus every problem found."""
  L = SCHEMA_LIMITS
  errs: list[str] = []
  if not isinstance(items, list) or len(items) < 2:
    return [], ["'keyframes' must be a list of at least two keyframes (start and end standing)"]
  kfs: list[KF] = []
  for i, it in enumerate(items):
    where = f"keyframe {i}"
    if not isinstance(it, dict):
      errs.append(f"{where} must be an object")
      continue
    t = _num(it.get("t"), f"{where}: t", errs)
    where = f"keyframe {i} (t={t:.2f})"
    body = np.zeros(6)
    feet = {}
    if it.get("pose") not in (None, "stand"):
      errs.append(f"{where}: pose can only be \"stand\"")
    if it.get("pose") != "stand":
      b = it.get("body", {}) or {}
      if not isinstance(b, dict):
        errs.append(f"{where}: body must be an object")
        b = {}
      for j, key in enumerate(("x", "y", "dz", "roll", "pitch", "yaw")):
        body[j] = _num(b.get(key, 0.0), f"{where}: body.{key}", errs)
      for key in b:
        if key not in ("x", "y", "dz", "roll", "pitch", "yaw"):
          errs.append(f"{where}: unknown body field {key!r} (use x, y, dz, roll, pitch, yaw)")
      f = it.get("feet", {}) or {}
      if not isinstance(f, dict):
        errs.append(f"{where}: feet must be an object")
        f = {}
      for leg, off in f.items():
        if leg not in LEGS:
          errs.append(f"{where}: unknown foot {leg!r}; feet are {list(LEGS)}")
          continue
        if not isinstance(off, (list, tuple)) or len(off) != 3:
          errs.append(f"{where}: feet.{leg} must be [dx, dy, dz]")
          continue
        feet[leg] = np.array([_num(v, f"{where}: feet.{leg}", errs) for v in off])
    ease = it.get("ease", "stop")
    if ease not in ("stop", "flow"):
      errs.append(f"{where}: ease must be \"stop\" or \"flow\"")
      ease = "stop"
    # range checks
    if abs(body[0]) > L["body_xy"] or abs(body[1]) > L["body_xy"]:
      errs.append(f"{where}: body x/y must stay within +-{L['body_xy']} m of the start")
    if not L["dz"][0] <= body[2] <= L["dz"][1]:
      errs.append(f"{where}: body dz must be in [{L['dz'][0]}, {L['dz'][1]}] m")
    for j, key in ((3, "roll"), (4, "pitch"), (5, "yaw")):
      if abs(body[j]) > L[key]:
        errs.append(f"{where}: {key} must be within +-{L[key]} deg")
    for leg, off in feet.items():
      if abs(off[0]) > L["foot_xy"] or abs(off[1]) > L["foot_xy"]:
        errs.append(f"{where}: feet.{leg} dx/dy must be within +-{L['foot_xy']} m")
      if not L["foot_dz"][0] <= off[2] <= L["foot_dz"][1]:
        errs.append(f"{where}: feet.{leg} dz must be in [0, {L['foot_dz'][1]}] m (0 = on the ground)")
    kfs.append(KF(t, body, feet, ease, i))
    hold = it.get("hold")
    if hold is not None:
      h = _num(hold, f"{where}: hold", errs)
      if h < 0:
        errs.append(f"{where}: hold must be >= 0")
      elif h > 0:
        kfs[-1].ease = "stop"
        kfs.append(KF(t + h, body.copy(), {k: v.copy() for k, v in feet.items()}, "stop", i))
  if errs:
    return kfs, errs
  for a, b in zip(kfs, kfs[1:]):
    if b.t - a.t < L["min_gap"] - 1e-9:
      errs.append(
        f"keyframe {b.src} (t={b.t:.2f}) comes {b.t - a.t:.2f} s after the previous one"
        f" (hold included); keyframes must be at least {L['min_gap']} s apart and in time order"
      )
  if abs(kfs[0].t) > 1e-9:
    errs.append("the first keyframe must be at t = 0")
  if kfs[-1].t > L["t_max"]:
    errs.append(f"the motion lasts {kfs[-1].t:.1f} s; keep it under {L['t_max']} s")
  for kf, name in ((kfs[0], "first"), (kfs[-1], "last")):
    if np.abs(kf.body[2:]).max() > 1e-6 or np.abs(kf.body[:2]).max() > 0.005 or any(
      np.abs(o).max() > 1e-6 for o in kf.feet.values()
    ):
      errs.append(f"the {name} keyframe must be {{\"pose\": \"stand\"}}: start and end standing")
  return kfs, errs


def _quintic(s):
  """Quintic Hermite basis with zero acceleration at both ends."""
  s2, s3 = s * s, s * s * s
  s4, s5 = s3 * s, s3 * s2
  return (
    1 - 10 * s3 + 15 * s4 - 6 * s5,  # p0
    s - 6 * s3 + 8 * s4 - 3 * s5,  # v0 * h
    10 * s3 - 15 * s4 + 6 * s5,  # p1
    -4 * s3 + 7 * s4 - 3 * s5,  # v1 * h
  )


def keyframe_track(kfs: list[KF], fps: float = FPS) -> tuple[np.ndarray, np.ndarray]:
  """Frame times and the 18 interpolated channels (T, 18).

  "stop" keyframes get zero velocity (exactly the minimum-jerk curve that
  md.build_motion uses); "flow" keyframes get a monotone Catmull-Rom velocity,
  so the motion passes through them without stopping and never overshoots.
  """
  t = np.array([k.t for k in kfs])
  V = np.stack([k.channels() for k in kfs])
  M = np.zeros_like(V)
  for i in range(1, len(kfs) - 1):
    if kfs[i].ease != "flow":
      continue
    d0 = (V[i] - V[i - 1]) / (t[i] - t[i - 1])
    d1 = (V[i + 1] - V[i]) / (t[i + 1] - t[i])
    m = 0.5 * (d0 + d1)
    m = np.where(d0 * d1 <= 0, 0.0, m)  # a local extremum: stop in that channel
    lim = 3.0 * np.minimum(np.abs(d0), np.abs(d1))
    M[i] = np.clip(m, -lim, lim)
  n = int(round(t[-1] * fps)) + 1
  times = np.arange(n) / fps
  out = np.zeros((n, V.shape[1]))
  for i, ti in enumerate(times):
    j = min(int(np.searchsorted(t, ti, side="right")) - 1, len(kfs) - 2)
    h = t[j + 1] - t[j]
    s = float(np.clip((ti - t[j]) / h, 0.0, 1.0))
    b0, b1, b2, b3 = _quintic(s)
    out[i] = b0 * V[j] + b1 * h * M[j] + b2 * V[j + 1] + b3 * h * M[j + 1]
  out[:, 8::3] = np.maximum(out[:, 8::3], md.FOOT_RADIUS)  # feet z never below the ground
  return times, out


def build_from_keyframes(kfs: list[KF], interp: str = "spline") -> md.Motion:
  """Keyframes -> a 50 Hz Pupper motion with IK (md.build_motion does the IK).

  interp="minjerk": md.build_motion on the keyframes, as in the Tricks notebook
  (the robot stops at every keyframe). interp="spline": the track above, sampled
  at every frame and handed to md.build_motion frame by frame, so "flow"
  keyframes really flow.
  """
  if interp == "minjerk":
    return md.build_motion([k.md_keyframe() for k in kfs], fps=FPS)
  times, track = keyframe_track(kfs)
  frames = [
    md.Keyframe(
      float(ti), tuple(c[0:3]), tuple(c[3:6]),
      feet={leg: tuple(c[6 + 3 * i: 9 + 3 * i]) for i, leg in enumerate(LEGS)},
    )
    for ti, c in zip(times, track)
  ]
  return md.build_motion(frames, fps=FPS)


def keyframe_errors(motion: md.Motion, kfs: list[KF]) -> list[dict]:
  """How far the built (or generated) motion is from each keyframe."""
  out = []
  for kf in kfs:
    i = min(int(round(kf.t * motion.fps)), len(motion.root_pos) - 1)
    k = kf.md_keyframe()
    got = KIN.feet(motion.root_pos[i], motion.root_quat[i], motion.joint_pos[i])
    foot = {leg: float(np.linalg.norm(got[leg] - np.array(k.feet[leg]))) for leg in LEGS}
    out.append(dict(
      t=kf.t, src=kf.src, foot_err=foot,
      root_err=float(np.linalg.norm(motion.root_pos[i] - np.array(k.root_pos))),
      rot_err=quat_angle_deg(motion.root_quat[i], md.rpy_to_quat(k.root_rpy)),
    ))
  return out


def reach_problems(motion: md.Motion, kfs: list[KF]) -> list[str]:
  """Keyframe targets the IK couldn't reach (> 2 mm)."""
  out = []
  for e in keyframe_errors(motion, kfs):
    leg, err = max(e["foot_err"].items(), key=lambda kv: kv[1])
    if err > 0.002:
      out.append(f"keyframe {e['src']} (t={e['t']:.2f}): foot {leg} can't reach its target"
                 f" (off by {100 * err:.1f} cm): move it closer or change the body pose")
  return out
