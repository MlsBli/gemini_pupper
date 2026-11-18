"""Augmentation on keyframes: mirror, time scale, jitter.

New in Phase 2. Augmenting the keyframes (not the joint angles) means the IK works
out the joints again, so a mirrored motion needs no abduction-sign bookkeeping, and
every augmented motion goes back through the same parse -> build -> gate path as
any other. Each function takes and returns a list of KF (keyframes.py) and leaves
its input unchanged; check() says whether the result is still a valid keyframe list.
"""
import numpy as np

from .keyframes import KF, parse_keyframes

BODY_KEYS = ("x", "y", "dz", "roll", "pitch", "yaw")
# Mirroring across the robot's centre plane (x-z): y, roll and yaw change sign
# (+y is left, roll > 0 raises the left side, yaw > 0 turns left), and left and
# right feet swap places with their sideways offset negated.
MIRROR_BODY_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
MIRROR_LEG = {"front_r": "front_l", "front_l": "front_r", "back_r": "back_l", "back_l": "back_r"}


def _copy(k: KF, **changes) -> KF:
  fields = dict(t=k.t, body=k.body.copy(), feet={leg: v.copy() for leg, v in k.feet.items()},
                ease=k.ease, src=k.src)
  fields.update(changes)
  return KF(**fields)


def mirror(kfs: list[KF]) -> list[KF]:
  """The same motion, left and right swapped."""
  out = []
  for k in kfs:
    feet = {MIRROR_LEG[leg]: v * np.array([1.0, -1.0, 1.0]) for leg, v in k.feet.items()}
    out.append(_copy(k, body=k.body * MIRROR_BODY_SIGN, feet=feet))
  return out


def time_scale(kfs: list[KF], factor: float) -> list[KF]:
  """The same poses, played `factor` times slower (0.8 = 25% faster)."""
  return [_copy(k, t=k.t * factor) for k in kfs]


def jitter(kfs: list[KF], rng: np.random.Generator,
           body_sigma=(0.003, 0.003, 0.003, 2.0, 2.0, 2.0), foot_sigma: float = 0.003) -> list[KF]:
  """Small random changes to every keyframe except the first and last (the stands).

  body_sigma: standard deviations for x, y, dz (m) and roll, pitch, yaw (deg).
  Only lifted feet (dz > 0) move; feet on the ground stay exactly where they are,
  and a lifted foot never goes below the ground.
  """
  out = [_copy(kfs[0])]
  for k in kfs[1:-1]:
    body = k.body + rng.normal(0.0, body_sigma)
    feet = {}
    for leg, v in k.feet.items():
      if v[2] > 0:
        v = v + rng.normal(0.0, foot_sigma, 3)
        v[2] = max(v[2], 0.005)
      feet[leg] = v.copy()
    out.append(_copy(k, body=body, feet=feet))
  out.append(_copy(kfs[-1]))
  return out


def to_items(kfs: list[KF]) -> list[dict]:
  """Keyframes back in the JSON format Gemini writes (readable, and parse_keyframes checks it)."""
  items = []
  for k in kfs:
    item = {"t": round(float(k.t), 4),
            "body": {key: round(float(v), 5) for key, v in zip(BODY_KEYS, k.body) if abs(v) > 1e-9}}
    if k.feet:
      item["feet"] = {leg: [round(float(x), 5) for x in v] for leg, v in k.feet.items()}
    item["ease"] = k.ease
    items.append(item)
  return items


def from_record(items: list[dict]) -> list[KF]:
  """Keyframes from a pipeline record's "keyframes" (t, body list, feet lists, ease)."""
  return [KF(float(k["t"]), np.array(k["body"], float), {leg: np.array(v, float) for leg, v in k["feet"].items()},
             k.get("ease", "stop")) for k in items]


def check(kfs: list[KF]) -> tuple[list[KF], list[str]]:
  """Round-trip through parse_keyframes: the augmented keyframes, and any problems
  (e.g. keyframes closer than 0.08 s after speeding up, or a pose out of range)."""
  return parse_keyframes(to_items(kfs))
