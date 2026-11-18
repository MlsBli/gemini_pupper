"""Quality metrics for a motion, and the benchmark's 12 requests with their precision checks.

Moved from the benchmark notebook: evaluate from cell "Core" (section "quality
metrics") and cell "The test requests, and how precision is measured", unchanged
except PHYS -> physics_gate() (see verify.py).
"""
import numpy as np

from mjlab.tasks.tracking.config.pupper import motion_design as md

from .motion import FPS, H, LEGS, feet_series, rpy_series
from .verify import CONTACT_Z, contact_analysis, physics_gate, slide_stats


def evaluate(motion: md.Motion, phys: dict | None = None) -> dict:
  """Objective numbers for one final motion. Lower is better unless noted."""
  rep = md.check_motion(motion)
  c = contact_analysis(motion)
  s = slide_stats(c["feet"], c["on"])
  dt = 1.0 / motion.fps
  q = motion.joint_pos
  acc = np.diff(q, 2, axis=0) / dt**2
  jerk = np.diff(q, 3, axis=0) / dt**3
  rpy = rpy_series(motion)
  lift = (c["feet"][..., 2] - md.FOOT_RADIUS).max(0)
  floating = (~c["on"].any(1)) & (c["lowest_body"] > CONTACT_Z)
  phys = phys or physics_gate().replay(motion)
  m = c["support_margin"]
  return dict(
    duration_s=rep["duration_s"],
    max_joint_speed=rep["max_joint_speed_rad_s"],
    joint_limit_frac=rep["fraction_of_joint_samples_at_limit"],
    lowest_foot_cm=round(100 * rep["lowest_foot_bottom_m"], 2),
    lowest_body_cm=round(100 * rep["lowest_body_point_m"], 2),
    slide_mean_cm_s=s["slide_mean_cm_s"],
    slide_worst_cm=s["slide_worst_cm"],
    floating_frac=round(float(floating.mean()), 3),
    acc_rms=round(float(np.sqrt((acc**2).mean())), 1),  # rad/s^2
    jerk_rms=round(float(np.sqrt((jerk**2).mean())), 0),  # rad/s^3: smoothness
    com_outside_frac=round(float(np.mean(np.nan_to_num(m, nan=-1) < 0)), 3),
    min_support_margin_cm=round(100 * float(np.nanmin(m)), 2) if np.isfinite(m).any() else None,
    # how much it moves (higher = more motion; catches "does nothing")
    joint_travel_rad=round(float(np.abs(np.diff(q, axis=0)).sum()), 1),
    body_height_range_cm=round(100 * float(np.ptp(motion.root_pos[:, 2])), 1),
    body_tilt_range_deg=round(float(max(np.ptp(rpy[:, 0]), np.ptp(rpy[:, 1]))), 1),
    body_yaw_range_deg=round(float(np.ptp(rpy[:, 2])), 1),
    max_foot_lift_cm=round(100 * float(lift.max()), 1),
    physics=phys["verdict"],
    physics_pos_cm=phys["worst_pos_cm"],
    physics_tilt_deg=phys["worst_tilt_deg"],
  )


def _feet_h(motion):
  return feet_series(motion)[..., 2] - md.FOOT_RADIUS  # (T, 4) foot bottom height


def spec_paw_lift(motion, leg, height, hold):
  h = _feet_h(motion)
  k = LEGS.index(leg)
  peak = float(h[:, k].max())
  held = float((h[:, k] > 0.7 * height).sum() / FPS)
  others = float(np.delete(h, k, axis=1).max())
  ok = abs(peak - height) <= 0.4 * height and abs(held - hold) <= 0.4 and others < 0.015
  return dict(ok=ok, measured=f"{leg} up {100 * peak:.1f} cm for {held:.2f} s; other feet up to {100 * others:.1f} cm",
              error=round(abs(peak - height) / height + abs(held - hold) / hold, 3))


def spec_body_drop(motion, depth, hold):
  d = H - motion.root_pos[:, 2]
  peak = float(d.max())
  held = float((d > 0.7 * depth).sum() / FPS)
  lift = float(_feet_h(motion).max())
  ok = abs(peak - depth) <= 0.4 * depth and abs(held - hold) <= 0.4 and lift < 0.015
  return dict(ok=ok, measured=f"down {100 * peak:.1f} cm for {held:.2f} s; feet up to {100 * lift:.1f} cm",
              error=round(abs(peak - depth) / depth + abs(held - hold) / hold, 3))


def spec_lean_lr(motion, deg):
  r = rpy_series(motion)[:, 0]
  left, right = float(-r.min()), float(r.max())  # lean left = roll < 0
  order = int(np.argmin(r)) < int(np.argmax(r))
  ok = abs(left - deg) <= 0.4 * deg and abs(right - deg) <= 0.4 * deg and order
  return dict(ok=ok, measured=f"left {left:.0f} deg, right {right:.0f} deg, {'left first' if order else 'right first'}",
              error=round((abs(left - deg) + abs(right - deg)) / (2 * deg) + (0 if order else 1), 3))


def spec_yaw_lr(motion, deg):
  y = rpy_series(motion)[:, 2]
  left, right = float(y.max()), float(-y.min())  # look left = yaw > 0
  order = int(np.argmax(y)) < int(np.argmin(y))
  ok = abs(left - deg) <= 0.4 * deg and abs(right - deg) <= 0.4 * deg and order
  return dict(ok=ok, measured=f"left {left:.0f} deg, right {right:.0f} deg, {'left first' if order else 'right first'}",
              error=round((abs(left - deg) + abs(right - deg)) / (2 * deg) + (0 if order else 1), 3))


def spec_nod(motion, count, deg):
  p = rpy_series(motion)[:, 1]  # pitch > 0: nose down
  n, armed = 0, True
  for v in p:
    if armed and v > 0.6 * deg:
      n, armed = n + 1, False
    elif v < 0.3 * deg:
      armed = True
  ok = n == count
  return dict(ok=ok, measured=f"{n} nods (nose down past {0.6 * deg:.0f} deg; max {p.max():.0f} deg)",
              error=round(abs(n - count) / count, 3))


SPECS = dict(paw_lift=spec_paw_lift, body_drop=spec_body_drop, lean_lr=spec_lean_lr,
             yaw_lr=spec_yaw_lr, nod=spec_nod)


REQUESTS = [
  # precise
  dict(id="paw", kind="precise", text="Lift your front left paw about 5 cm off the ground, hold it up for one second, then put it back down.",
       spec=("paw_lift", dict(leg="front_l", height=0.05, hold=1.0))),
  dict(id="crouch", kind="precise", text="Crouch down about 4 cm lower than normal, hold it for a second, then stand back up.",
       spec=("body_drop", dict(depth=0.04, hold=1.0))),
  dict(id="lean", kind="precise", text="Lean your body to the left about 15 degrees, then to the right about 15 degrees, then back to the middle.",
       spec=("lean_lr", dict(deg=15))),
  dict(id="nod", kind="precise", text="Nod your body down and up twice, like you're saying yes.",
       spec=("nod", dict(count=2, deg=10))),
  dict(id="look", kind="precise", text="Keep your feet planted and turn your body to look left about 20 degrees, then right about 20 degrees.",
       spec=("yaw_lr", dict(deg=20))),
  # style
  dict(id="bow", kind="style", text="Do a play bow, like you want to play."),
  dict(id="shake", kind="style", text="Shake your body like a wet dog."),
  dict(id="sit", kind="style", text="Sit down like a good dog."),
  dict(id="stretch", kind="style", text="Stretch like you just woke up from a nap."),
  dict(id="excited", kind="style", text="Act excited to see me!"),
  dict(id="sad", kind="style", text="Look sad."),
  # sequence
  dict(id="bow_wiggle", kind="sequence", text="Bow down, then pop back up and wiggle your hips."),
]


def measure_spec(req, motion):
  if not req.get("spec"):
    return None
  name, kw = req["spec"]
  return SPECS[name](motion, **kw)
