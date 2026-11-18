"""Request in, gated motion out: keyframes from the LLM, built, verified, retried.

Moved from the benchmark notebook, cell "The three pipelines", option 1 only (the
UniMate options stay in the notebook). Changes: Config keeps only option 1's fields;
run_keyframes takes the LLM to use (the notebook used a global `llm`) and calls
keyframe_system() (see prompt.py); request_to_motion is a small wrapper for a bare
request string.
"""
import dataclasses

import numpy as np

from .keyframes import build_from_keyframes, keyframe_errors, parse_keyframes, reach_problems
from .llm import LLM, Stopwatch
from .metrics import evaluate, measure_spec
from .motion import motion_to_dict
from .prompt import keyframe_system, parse_json
from .verify import finalize, gate


@dataclasses.dataclass
class Config:
  kf_interp: str = "spline"  # "spline" (stop/flow keyframes) or "minjerk" (md.build_motion as in the Tricks notebook)
  kf_max_attempts: int = 3  # LLM calls, counting retries after failed checks


CFG = Config()


def feedback(user: str, text: str, errs: list[str]) -> str:
  return (
    f"{user}\n\nYour previous answer:\n{text.strip()}\n\nIt failed these checks:\n"
    + "\n".join(f"- {e}" for e in errs)
    + "\n\nFix every problem and reply with the complete corrected JSON object only."
  )


def _record(option, req, sw, motion, g, **extra) -> dict:
  rec = dict(
    option=option, request_id=req["id"], request=req["text"], kind=req["kind"],
    ok=bool(g["ok"]) if g else False, reasons=g["reasons"] if g else ["no motion"],
    latency=dict(sw.stages), total_s=sw.total(), llm_calls=list(sw.llm_calls),
    motion=motion_to_dict(motion) if motion is not None else None,
    replay=g["physics"]["traj"] if g else None,
    metrics=evaluate(motion, g["physics"]) if motion is not None else None,
    spec=measure_spec(req, motion) if motion is not None else None,
  )
  rec.update(extra)
  return rec


def _kf_adherence(motion, kfs):
  if motion is None or not kfs:
    return None
  e = keyframe_errors(motion, kfs)
  return dict(
    foot_cm_mean=round(100 * float(np.mean([v for x in e for v in x["foot_err"].values()])), 2),
    foot_cm_max=round(100 * float(max(v for x in e for v in x["foot_err"].values())), 2),
    root_cm_max=round(100 * float(max(x["root_err"] for x in e)), 2),
    rot_deg_max=round(float(max(x["rot_err"] for x in e)), 1),
  )


def _kfs_json(kfs):
  return [dict(t=k.t, body=k.body.tolist(), feet={l: v.tolist() for l, v in k.feet.items()}, ease=k.ease) for k in kfs]


# ---------------------------------------------------------------------------- option 1
def run_keyframes(req, llm: LLM, cfg: Config = CFG) -> dict:
  sw = Stopwatch(llm)
  user = f'Request: "{req["text"]}"'
  attempts, best = [], None
  for _ in range(cfg.kf_max_attempts):
    text = sw.llm(keyframe_system(), user)
    errs, kfs, motion, g = [], [], None, None
    try:
      data = parse_json(text)
      kfs, errs = parse_keyframes(data.get("keyframes"))
    except ValueError as e:
      errs = [str(e)]
    if not errs:
      with sw("build"):
        motion = build_from_keyframes(kfs, cfg.kf_interp)
      with sw("verify"):
        errs = reach_problems(motion, kfs)
        g = gate(motion)
        errs += g["reasons"]
    attempts.append(dict(text=text, errors=errs))
    if motion is not None and (best is None or len(errs) < len(best[3])):
      best = (motion, kfs, g, errs)
    if not errs:
      break
    user = feedback(f'Request: "{req["text"]}"', text, errs)
  if best is None:
    return _record("1_keyframes", req, sw, None, None, attempts=attempts)
  motion, kfs, g, errs = best
  built = motion
  with sw("finalize"):
    motion, pre, slow = finalize(motion)
  if pre or slow != 1.0:
    with sw("gate"):
      g = gate(motion)
  if errs and g["ok"]:  # IK-reach problems aren't in the gate; keep them as a failure
    g = dict(g, ok=False, reasons=errs)
  return _record("1_keyframes", req, sw, motion, g, attempts=attempts, keyframes=_kfs_json(kfs),
                 kf_adherence=_kf_adherence(built, kfs), pre_frames=pre, slow=slow)


def request_to_motion(text: str, llm: LLM, cfg: Config = CFG, request_id: str = "request") -> dict:
  """One spoken request -> the benchmark's record for it (run_keyframes on a bare string).

  The record's "ok" says whether the final motion passed; "motion" holds its arrays
  (motion_from_dict turns them back into an md.Motion); "reasons" says what failed.
  """
  return run_keyframes(dict(id=request_id, text=text, kind="request"), llm, cfg)
