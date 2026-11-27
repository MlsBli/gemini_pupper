"""The zero-shot demo: a typed request in, a video of the general tracker playing it out.

New for the TA demo notebook (demo_zero_shot.ipynb). The whole chain, in simulation:

  1. Gemini writes the motion as keyframes; the gate checks it (kflib.pipeline)
  2. the general tracker plays it (pupper-mjlab's simulation, kflib.track), and so
     does plain open-loop replay, for comparison
  3. a side-by-side video of the reference, plain replay and the tracker, drawn from
     the trajectories the robot actually followed (pupper-mjlab's motion renderer)

The tracker never trained on the motion: it was written a few seconds earlier.
Rendering needs MUJOCO_GL=egl on Colab (set before mujoco is first imported).
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def paw_lift(reference, tracked) -> dict | None:
  """The biggest paw lift in the reference, and how much of it the robot did (None: no lift)."""
  from mjlab.tasks.tracking.config.pupper import motion_design as md

  from .motion import LEGS, feet_series

  ref = feet_series(reference)[..., 2] - md.FOOT_RADIUS
  k = int(np.argmax(ref.max(0)))
  if ref[:, k].max() < 0.02:
    return None
  got = feet_series(tracked)[..., 2] - md.FOOT_RADIUS
  return dict(foot=LEGS[k], asked_cm=round(100 * float(ref[:, k].max()), 1), done_cm=round(100 * float(got[:, k].max()), 1))


def render(motions: list, labels: list[str], out_mp4, width: int = 360, height: int = 300) -> Path:
  """Motions side by side, labelled, to an mp4 at 50 fps."""
  import mediapy

  from mjlab.tasks.tracking.config.pupper import unimate as U

  video = U.render([SimpleNamespace(motion=m) for m in motions], labels, width=width, height=height, skeleton=False)
  mediapy.write_video(str(out_mp4), video, fps=50)
  return Path(out_mp4)


def run(request: str, llm, checkpoint, out_dir) -> dict:
  """The whole chain for one request. Returns what happened, and the video's path."""
  from .motion import make_motion, motion_from_dict
  from .pipeline import request_to_motion
  from .prompt import parse_json
  from .track import track

  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  rec = request_to_motion(request, llm)
  plan = ""
  try:
    plan = str(parse_json(rec["attempts"][-1]["text"]).get("plan", ""))
  except (ValueError, KeyError, IndexError, TypeError):
    pass
  result = dict(request=request, plan=plan, attempts=len(rec.get("attempts") or []), gate_ok=rec["ok"],
                reasons=rec["reasons"])
  if rec["motion"] is None:
    return result
  reference = motion_from_dict(rec["motion"])
  npz = reference.save_npz(out / "motion.npz")
  result.update(npz=str(npz), duration_s=round(reference.duration, 1))
  played = {}
  tracker_mode = "json" if str(checkpoint).endswith(".json") else "checkpoint"  # a .pt, or an exported policy.json
  for label, mode, policy in (("general tracker", tracker_mode, checkpoint), ("plain replay", "plain", None)):
    r = track(npz, mode, policy)
    t = r.pop("tracked")
    played[label] = make_motion(t["root_pos"].astype(float), t["root_quat"].astype(float), t["joint_pos"].astype(float))
    result[label] = dict(r, paw=paw_lift(reference, played[label]))

  def tag(label):
    r = result[label]
    return f"{label}: {r['verdict']} ({r['pos_cm']} cm, {r['tilt_deg']} deg)"

  result["video"] = str(render([reference, played["plain replay"], played["general tracker"]],
                               ["reference (Gemini's keyframes)", tag("plain replay"), tag("general tracker")],
                               out / "side_by_side.mp4"))
  return result


def describe(result: dict) -> str:
  """A few readable lines about one run (for the notebook)."""
  lines = [f"request: {result['request']}"]
  if result.get("plan"):
    lines.append(f"Gemini's plan: {result['plan']}")
  lines.append(f"keyframes: {result['attempts']} attempt(s); the gate "
               + ("passed" if result["gate_ok"] else "did not pass: " + "; ".join(result["reasons"])[:300]))
  for label in ("general tracker", "plain replay"):
    r = result.get(label)
    if r:
      paw = r.get("paw")
      lines.append(f"{label:16} {r['verdict']:9} worst {r['pos_cm']} cm, {r['tilt_deg']} deg"
                   + (f" | {paw['foot']} paw up {paw['done_cm']} of {paw['asked_cm']} cm" if paw else ""))
  return "\n".join(lines)
