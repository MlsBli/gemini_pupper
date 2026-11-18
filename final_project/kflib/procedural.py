"""Procedural motions: random but sensible keyframe lists, in the format Gemini writes.

New in Phase 2. Language-made motions cluster around a few kinds of motion (bows,
wiggles, paw lifts); these fill in the joint space more evenly. Each motion is a
list of moves between two stands, written as Gemini's JSON keyframes (so it's
readable, and parse_keyframes / build_from_keyframes / gate treat it like any other
motion). The same seed always gives the same keyframes.

Moves:
  pose    1-3 of the body's dz, pitch, roll, yaw, x, y at once, inside a fraction
          of the reach measured by probe_limits()
  lift    one paw, with the lean-away rule from the robot brief: shift the body
          about 2 cm away from the paw first, lift it, maybe wave it, put it back
          exactly where it stood, shift back
  wiggle  2-4 quick alternating poses in one direction, "flow" (passing through)
After each move Pupper sometimes returns to the stand. Timing, holds and stop/flow
are random. The gate decides what's kept; nothing here guarantees a pass.
"""
import dataclasses

import numpy as np

from .motion import LEGS
from .prompt import reach_limits


@dataclasses.dataclass
class ProcConfig:
  n_moves: tuple[int, int] = (2, 5)  # moves per motion (inclusive)
  move_s: tuple[float, float] = (0.3, 1.0)  # seconds to reach a pose
  reach_frac: float = 0.75  # use this fraction of each measured range (combined poses reach less)
  p_lift: float = 0.15  # chance a move is a paw lift (0.3 in the first v1: 78% of motions lifted a paw)
  p_wiggle: float = 0.15  # chance a move is a wiggle
  p_hold: float = 0.35  # chance a pose is held
  hold_s: tuple[float, float] = (0.2, 0.8)
  p_flow: float = 0.3  # chance a pose is passed through ("flow") instead of stopped at
  p_back_to_stand: float = 0.3  # chance of returning to the stand between moves
  lift_dz: tuple[float, float] = (0.02, 0.06)  # paw lift height (m)
  max_start_s: float = 6.5  # no new move starts after this (motions stay under the 10 s limit)
  # How often a pose move uses each body channel (it picks 1-3, weighted). Height is
  # favoured: with equal weights, real crouches were rare (median height range 1 cm).
  pose_weights: tuple[float, ...] = (0.3, 0.2, 0.15, 0.15, 0.1, 0.1)  # POSE_KEYS order


POSE_KEYS = ("dz", "pitch", "roll", "yaw", "x", "y")
ROUND = {"dz": 3, "x": 3, "y": 3, "pitch": 1, "roll": 1, "yaw": 1}


def _lean_away(leg: str) -> dict:
  """Body shift before lifting `leg` (the robot brief: lifting front_l -> x -0.015, y -0.02)."""
  front, left = leg.startswith("front"), leg.endswith("_l")
  return {"x": -0.015 if front else 0.015, "y": -0.02 if left else 0.02, "dz": -0.01}


def random_keyframes(seed: int, cfg: ProcConfig = ProcConfig()) -> list[dict]:
  """One random motion as Gemini-format keyframes, from the stand back to the stand."""
  rng = np.random.default_rng(seed)
  lim = reach_limits()
  items = [{"t": 0.0, "pose": "stand"}]
  t = 0.0

  def add(t_new, body=None, feet=None, ease="stop", hold=None):
    item = {"t": round(float(t_new), 3)}
    if body:
      item["body"] = {k: round(float(v), ROUND[k]) for k, v in body.items()}
    if feet:
      item["feet"] = {leg: [round(float(x), 3) for x in v] for leg, v in feet.items()}
    if ease != "stop":
      item["ease"] = ease
    if hold:
      item["hold"] = round(float(hold), 2)
      t_new += item["hold"]
    items.append(item)
    return t_new

  def pose_value(key):
    lo, hi = lim[key]
    return rng.uniform(cfg.reach_frac * lo, cfg.reach_frac * hi)

  def maybe_hold():
    return rng.uniform(*cfg.hold_s) if rng.random() < cfg.p_hold else None

  for _ in range(rng.integers(cfg.n_moves[0], cfg.n_moves[1] + 1)):
    if t > cfg.max_start_s:
      break
    kind = rng.choice(["pose", "lift", "wiggle"], p=[1 - cfg.p_lift - cfg.p_wiggle, cfg.p_lift, cfg.p_wiggle])
    if kind == "pose":
      w = np.array(cfg.pose_weights, float)
      keys = rng.choice(POSE_KEYS, size=rng.integers(1, 4), replace=False, p=w / w.sum())
      ease = "flow" if rng.random() < cfg.p_flow else "stop"
      t = add(t + rng.uniform(*cfg.move_s), {k: pose_value(k) for k in keys}, ease=ease,
              hold=None if ease == "flow" else maybe_hold())
    elif kind == "lift":
      leg = LEGS[rng.integers(len(LEGS))]
      lean = _lean_away(leg)
      t = add(t + rng.uniform(0.4, 0.8), lean)  # shift away first, feet still planted
      up = [rng.uniform(-0.03, 0.03), rng.uniform(-0.01, 0.01), rng.uniform(*cfg.lift_dz)]
      t = add(t + rng.uniform(0.3, 0.6), lean, {leg: up}, hold=maybe_hold())
      if rng.random() < 0.4:  # a wave: the paw moves while it's up
        wave = [up[0] + rng.uniform(-0.03, 0.03), up[1], up[2] + rng.uniform(-0.01, 0.01)]
        t = add(t + rng.uniform(0.25, 0.5), lean, {leg: [wave[0], wave[1], max(wave[2], 0.02)]}, ease="flow")
      t = add(t + rng.uniform(0.3, 0.6), lean)  # down, exactly where it stood
      t = add(t + rng.uniform(0.3, 0.6), None, ease="stop")  # shift back to the stand
    else:  # wiggle
      key = rng.choice(["roll", "yaw", "pitch", "y"])
      amp = abs(pose_value(key)) * rng.uniform(0.4, 0.8)
      sign = rng.choice([-1.0, 1.0])
      for _ in range(rng.integers(2, 5)):
        t = add(t + rng.uniform(0.2, 0.4), {key: sign * amp}, ease="flow")
        sign = -sign
      t = add(t + rng.uniform(0.25, 0.45), None, ease="stop")
    if rng.random() < cfg.p_back_to_stand and kind == "pose":
      t = add(t + rng.uniform(*cfg.move_s), None)
  items.append({"t": round(t + rng.uniform(*cfg.move_s), 3), "pose": "stand"})
  return items
