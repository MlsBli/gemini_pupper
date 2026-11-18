"""What the LLM is told: Pupper's measured reach, the robot brief, the keyframe format,
two worked examples and the system prompt; plus parse_json for its replies.

Moved from the benchmark notebook, cell "What the LLM is told". The text is unchanged.
One change: the notebook measured the reach (LIM) and built ROBOT_BRIEF and
KEYFRAME_SYSTEM when the cell ran; here robot_brief() and keyframe_system() build the
same strings the first time they are needed, so importing kflib is quick. The
UniMate prompts (options 2 and 3) stay in the notebook.
"""
import functools
import json
import re

import numpy as np

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import DEFAULT_POSE
from mjlab.tasks.tracking.config.pupper import motion_design as md

from .keyframes import KF
from .motion import H, KIN, LEGS, STAND_FEET


def _pose_ok(body=(0, 0, 0, 0, 0, 0), feet=None) -> bool:
  """Can IK reach this pose with every target within 2 mm, nothing in the floor,
  no joint at a limit?"""
  kf = KF(0.0, np.array(body, float), feet or {}).md_keyframe()
  root, rq = np.array(kf.root_pos), md.rpy_to_quat(kf.root_rpy)
  q = np.array(DEFAULT_POSE, float)
  for leg in LEGS:
    q, err = KIN.solve_leg(leg, kf.feet[leg], root, rq, q, iters=200)
    if err > 0.002:
      return False
  KIN.set_state(root, rq, q)
  if KIN.lowest_body_point() < -0.001:
    return False
  return not np.any((q <= KIN.lower + 0.02) | (q >= KIN.upper - 0.02))


def _range(vals, test):
  """The contiguous range of vals around 0 that passes test."""
  vals = np.asarray(vals, float)
  ok = np.array([test(v) for v in vals])
  i0 = int(np.argmin(np.abs(vals)))
  lo = hi = i0
  while lo > 0 and ok[lo - 1]:
    lo -= 1
  while hi < len(vals) - 1 and ok[hi + 1]:
    hi += 1
  return float(vals[lo]), float(vals[hi])


def probe_limits() -> dict:
  L = {}
  L["dz"] = _range(np.arange(-0.12, 0.0501, 0.005), lambda v: _pose_ok((0, 0, v, 0, 0, 0)))
  L["x"] = _range(np.arange(-0.08, 0.0801, 0.005), lambda v: _pose_ok((v, 0, -0.02, 0, 0, 0)))
  L["y"] = _range(np.arange(-0.08, 0.0801, 0.005), lambda v: _pose_ok((0, v, -0.02, 0, 0, 0)))
  L["pitch"] = _range(np.arange(-50, 50.1, 2.5), lambda v: _pose_ok((0, 0, -0.03, 0, v, 0)))
  L["roll"] = _range(np.arange(-40, 40.1, 2.5), lambda v: _pose_ok((0, 0, -0.03, v, 0, 0)))
  L["yaw"] = _range(np.arange(-50, 50.1, 2.5), lambda v: _pose_ok((0, 0, -0.03, 0, 0, v)))
  lean = (-0.015, -0.02, -0.01, 0, 0, 0)  # body shifted back-right, as when lifting front_l
  L["lift"] = _range(np.arange(0.0, 0.1201, 0.005),
                     lambda v: _pose_ok(lean, {"front_l": np.array([0, 0, v])}))
  L["lift_dx"] = _range(np.arange(-0.08, 0.0801, 0.005),
                        lambda v: _pose_ok(lean, {"front_l": np.array([v, 0, 0.03])}))
  return L


@functools.cache
def reach_limits() -> dict:
  """probe_limits(), measured once (a few seconds) and then remembered (the notebook's LIM)."""
  return probe_limits()


@functools.cache
def robot_brief() -> str:
  """The robot description the LLM gets (the notebook's ROBOT_BRIEF), with measured reach."""
  LIM = reach_limits()
  _sf = {leg: STAND_FEET[leg][:2].round(3).tolist() for leg in LEGS}
  return f"""\
## Pupper
A small robot dog: 2-3 kg, about 26 cm long, 12 motors (each leg: abduction, hip, knee). No head,
no tail, no neck: everything expressive comes from the body pose and the legs. Standing, its
body (base_link) is {H:.3f} m above the ground.

## Coordinates (metres, seconds, degrees)
x forward, y left, z up. The robot starts standing at x = y = 0, facing +x.
- body: x, y = body position (m) relative to the start; dz = height change from standing
  (negative = lower); roll, pitch, yaw in degrees.
  pitch > 0 lowers the nose (head down), pitch < 0 raises it.
  roll > 0 raises the left side (leans right); roll < 0 leans left.
  yaw > 0 turns left.
- feet: front_r, front_l, back_r, back_l. Each foot's offset [dx, dy, dz] (m, world frame) from
  where it stands. dz = 0 is on the ground, dz > 0 lifts it. Feet you leave out stay planted.
  Standing foot positions (x, y): {_sf}

## What Pupper can reach (measured with its inverse kinematics; one thing at a time)
- body dz {LIM['dz'][0]:+.3f} to {LIM['dz'][1]:+.3f}; body x {LIM['x'][0]:+.3f} to {LIM['x'][1]:+.3f}; body y {LIM['y'][0]:+.3f} to {LIM['y'][1]:+.3f} (all feet down)
- pitch {LIM['pitch'][0]:+.0f} to {LIM['pitch'][1]:+.0f}, roll {LIM['roll'][0]:+.0f} to {LIM['roll'][1]:+.0f}, yaw {LIM['yaw'][0]:+.0f} to {LIM['yaw'][1]:+.0f} (feet planted, body 3 cm low)
- a lifted front foot: dz up to {LIM['lift'][1]:.3f}, dx {LIM['lift_dx'][0]:+.3f} to {LIM['lift_dx'][1]:+.3f} (at dz 0.03)
Combining several of these at once reaches less; stay well inside the ranges.

## What gets checked before the robot moves (a failed check means the motion is rejected)
- Balance: a 2 kg robot with small round feet. The centre of mass must stay over the feet on
  the ground; it may leave them only briefly (under 0.3 s, e.g. a small hop). Keep 3 feet down.
  Before lifting a foot, shift the body about 2 cm away from it (lifting front_l: body x -0.015,
  y -0.02) and keep that shift while the foot is up. Never lift two feet on the same side.
- Physics: the motion is replayed in a physics simulator; the robot must not fall over.
- Speed: joints stay under 15 rad/s. A big move needs at least 0.3 s; typical moves 0.3-1.0 s.
- No sliding: a foot on the ground must not move sideways. To move a foot, lift it (dz >= 0.02)
  on the way. Nothing may go into the ground.
"""


KEYFRAME_FORMAT = """\
## Keyframe format (JSON)
{"t": 0.0, "pose": "stand"}
{"t": 0.6, "body": {"x": 0, "y": 0, "dz": -0.02, "roll": 0, "pitch": 10, "yaw": 0},
 "feet": {"front_l": [0.0, 0.0, 0.04]}, "ease": "stop", "hold": 0.5}
- t: seconds, starting at 0, increasing, at least 0.1 s apart (a hold counts as time spent).
- body fields default to 0; feet you leave out stay where they stand.
- ease: "stop" (default) = come to rest at this keyframe. "flow" = pass through it without
  stopping (for continuous moves: wiggles, waves, a nod in a series).
- hold: optional seconds to stay in this pose before moving on.
- The first keyframe is {"t": 0, "pose": "stand"} and the last is a stand too.
"""


EXAMPLE_SNEEZE = {
  "plan": "A dramatic sneeze: ah... ah... (nose rises, body leans back), CHOO (nose snaps down, "
          "body drops), a tiny hop on the front feet, settle to a stand.",
  "keyframes": [
    {"t": 0.0, "pose": "stand"},
    {"t": 0.3, "pose": "stand"},
    {"t": 0.75, "body": {"x": -0.01, "dz": 0.004, "pitch": -8}},
    {"t": 1.05, "body": {"x": -0.015, "dz": 0.006, "pitch": -10}},
    {"t": 1.23, "body": {"x": 0.01, "dz": -0.035, "pitch": 22}},
    {"t": 1.43, "body": {"dz": 0.012, "pitch": -6},
     "feet": {"front_r": [0, 0, 0.025], "front_l": [0, 0, 0.025]}},
    {"t": 1.57, "body": {"dz": -0.008, "pitch": 3}},
    {"t": 1.87, "pose": "stand"},
    {"t": 2.37, "pose": "stand"},
  ],
}


EXAMPLE_STEP = {
  "plan": "Step the right hind foot forward and back: lean forward-left first so the other three "
          "feet carry the weight, lift and place the foot, then reverse.",
  "keyframes": [
    {"t": 0.0, "pose": "stand"},
    {"t": 0.5, "pose": "stand"},
    {"t": 0.8, "body": {"x": 0.02, "y": 0.015, "dz": -0.005}},
    {"t": 1.0, "body": {"x": 0.02, "y": 0.015, "dz": -0.005}, "feet": {"back_r": [0.022, 0, 0.04]}},
    {"t": 1.2, "body": {"x": 0.02, "y": 0.015, "dz": -0.005}, "feet": {"back_r": [0.044, 0, 0]}, "hold": 0.5},
    {"t": 1.9, "body": {"x": 0.02, "y": 0.015, "dz": -0.005}, "feet": {"back_r": [0.022, 0, 0.04]}},
    {"t": 2.1, "body": {"x": 0.02, "y": 0.015, "dz": -0.005}},
    {"t": 2.5, "pose": "stand"},
  ],
}


@functools.cache
def keyframe_system() -> str:
  """The keyframe system prompt (the notebook's KEYFRAME_SYSTEM)."""
  return f"""\
You are the motion designer for Pupper, a small robot dog. Someone asked Pupper to do something.
You write the motion as keyframes; they are interpolated, solved with inverse kinematics, and
checked in a physics simulator before the robot moves. If a check fails you get the problems
back and fix them.

{robot_brief()}
{KEYFRAME_FORMAT}
## Designing a good motion
- Make it read as the request. Anticipation (a small opposite move first), a clear main action,
  a settle. Living things don't move at constant speed and don't stop dead between every pose:
  use "flow" for poses the body passes through, "stop"/"hold" for poses that should register.
- If the request gives numbers (cm, degrees, seconds, which paw, how many times), match them.
- 1.5 to 8 s in total.

Reply with only the JSON object {{"plan": "...", "keyframes": [...]}}: no prose, no code fences.

## Example: "sneeze"
{json.dumps(EXAMPLE_SNEEZE)}

## Example: "step your right hind foot forward and back"
{json.dumps(EXAMPLE_STEP)}
"""


def parse_json(text: str) -> dict:
  """The JSON object in an LLM reply (code fences and stray prose tolerated)."""
  t = text.strip()
  t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
  a, b = t.find("{"), t.rfind("}")
  if a < 0 or b < a:
    raise ValueError("the reply has no JSON object; reply with only the JSON object")
  try:
    obj = json.loads(t[a: b + 1])
  except json.JSONDecodeError as e:
    raise ValueError(f"the reply is not valid JSON ({e}); reply with only the JSON object") from e
  if not isinstance(obj, dict):
    raise ValueError("the reply must be one JSON object")
  return obj
