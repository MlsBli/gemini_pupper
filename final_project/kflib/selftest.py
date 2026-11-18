"""Self-tests for kflib: does each piece do what it claims?

    python -m kflib.selftest                       # parts 1-4 (CPU, no LLM, ~1 min)
    python -m kflib.selftest --runs <Drive>/pupper_generation_benchmark/runs
                                                   # + part 5: the regression against the benchmark

1. The benchmark's self-tests 5a, unchanged (the gate on motions we know).
2. Format errors come back as sentences Gemini can act on.
3. Round trips: motion dicts, and the npz the tracker loads.
4. Two keyframe motions join into a training library with no jump at the boundary
   (Phase 0, question 1: the tracker trains a library as one long motion).
5. Regression: every keyframe result the benchmark saved is rebuilt from its stored
   keyframes and must match: the motion, the gate verdict, the metrics, and the prompt.
Nothing here calls an LLM or moves the robot.
"""
import argparse
import math
import pickle
import tempfile
from pathlib import Path

import numpy as np

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.tricks import sit as course_sit

from .keyframes import FPS, KF, build_from_keyframes, parse_keyframes
from .metrics import evaluate
from .motion import motion_from_dict, motion_to_dict, quat_to_rpy
from .prompt import EXAMPLE_SNEEZE, EXAMPLE_STEP, keyframe_system
from .verify import finalize, gate

RESULTS = []


def _check(name, ok, detail=""):
  print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
  RESULTS.append(bool(ok))
  return ok


def part1_gate():
  print("--- 1. The gate on motions we know (the benchmark's 5a)")
  stand = md.build_motion([md.stand(0.0), md.stand(2.0)], fps=FPS)
  g = gate(stand)
  _check("a still stand passes", g["ok"] and g["physics"]["verdict"] == "HOLDS UP", g["physics"]["verdict"])
  g = gate(md.build_motion(course_sit.keyframes(), fps=FPS))
  _check("the course's sit (a trained trick) passes", g["ok"], f"{g['physics']['verdict']} {g['reasons']}")
  k, _ = parse_keyframes(EXAMPLE_SNEEZE["keyframes"])
  g = gate(build_from_keyframes(k))
  _check("the course's sneeze (a trained trick) passes", g["ok"], f"{g['physics']['verdict']} {g['reasons']}")
  k, _ = parse_keyframes([{"t": 0, "pose": "stand"},
                          {"t": 0.5, "feet": {"front_r": [0, 0, 0.06], "front_l": [0, 0, 0.06]}, "hold": 1.0},
                          {"t": 2.0, "pose": "stand"}])
  g = gate(build_from_keyframes(k))
  _check("both front paws up for 1 s is rejected", not g["ok"], g["reasons"][0][:90] if g["reasons"] else "")
  k, _ = parse_keyframes([{"t": 0, "pose": "stand"}, {"t": 0.15, "body": {"dz": -0.07, "pitch": 30}}, {"t": 0.3, "pose": "stand"}])
  g = gate(build_from_keyframes(k))
  _check("a too-fast move is rejected", not g["ok"], g["reasons"][0][:90] if g["reasons"] else "")


def part2_format():
  print("\n--- 2. Format errors are sentences Gemini can act on")
  for name, items, expect in [
    ("a motion that doesn't end standing",
     [{"t": 0, "pose": "stand"}, {"t": 1.0, "body": {"dz": -0.03}}], "last keyframe must be"),
    ("keyframes out of time order",
     [{"t": 0, "pose": "stand"}, {"t": 1.0, "body": {"dz": -0.03}}, {"t": 0.5, "pose": "stand"}],
     "in time order"),
    ("an unknown foot",
     [{"t": 0, "pose": "stand"}, {"t": 0.5, "feet": {"tail": [0, 0, 0.02]}}, {"t": 1.0, "pose": "stand"}],
     "unknown foot 'tail'"),
    ("a body pose far out of range",
     [{"t": 0, "pose": "stand"}, {"t": 0.5, "body": {"pitch": 80}}, {"t": 1.0, "pose": "stand"}],
     "pitch must be within"),
  ]:
    _, errs = parse_keyframes(items)
    hit = [e for e in errs if expect in e]
    _check(f"{name} is rejected with a reason", bool(hit), (hit or errs or ["no error"])[0][:100])
  for name, ex in (("sneeze", EXAMPLE_SNEEZE), ("step", EXAMPLE_STEP)):
    _, errs = parse_keyframes(ex["keyframes"])
    _check(f"the prompt's {name} example is valid", not errs, "; ".join(errs)[:100])


def _sneeze():
  k, _ = parse_keyframes(EXAMPLE_SNEEZE["keyframes"])
  return build_from_keyframes(k)


def part3_roundtrip():
  print("\n--- 3. Round trips")
  m = _sneeze()
  back = motion_from_dict(motion_to_dict(m))
  same = all(np.array_equal(getattr(m, a), getattr(back, a)) for a in ("root_pos", "root_quat", "joint_pos"))
  _check("motion_from_dict(motion_to_dict(m)) is identical", same)
  with tempfile.TemporaryDirectory() as d:
    path = m.save_npz(Path(d) / "sneeze.npz")
    with np.load(path) as z:
      keys = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
      T = len(m.root_pos)
      ok = (keys <= set(z.files) and float(z["fps"][0]) == 50.0 and z["joint_pos"].shape == (T, 12)
            and z["body_pos_w"].shape[:2] == (T, 13))
      _check("save_npz writes the tracker's arrays at 50 Hz (T frames, 12 joints, 13 bodies)", ok,
             f"fps {float(z['fps'][0])}, joint_pos {z['joint_pos'].shape}, body_pos_w {z['body_pos_w'].shape}")


def part4_joinable():
  print("\n--- 4. Keyframe motions join into a library with no jump")
  a = _sneeze()
  k, _ = parse_keyframes(EXAMPLE_STEP["keyframes"])
  b = build_from_keyframes(k)
  for name, m in (("sneeze", a), ("step", b)):
    for i, end in ((0, "starts"), (-1, "ends")):
      xy_mm = 1000 * float(np.linalg.norm(m.root_pos[i][:2]))
      yaw = abs(float(quat_to_rpy(m.root_quat[i])[2]))
      _check(f"{name} {end} at the origin, facing +x", xy_mm <= 5.0 and yaw <= 1.0, f"{xy_mm:.1f} mm, yaw {yaw:.2f} deg")
  try:
    lib = md.concat_clips([md.Clip("sneeze", a), md.Clip("step", b)])
    seam = len(a.root_pos) - 1  # the frame the two clips share
    jump_mm = 1000 * float(np.abs(np.diff(lib.root_pos[seam - 1:seam + 2], axis=0)).max())
    _check("md.concat_clips joins them, root moves under 5 mm a frame at the seam",
           jump_mm < 5.0 and len(lib.clips) == 2, f"{jump_mm:.2f} mm, clips {[c['name'] for c in lib.clips]}")
  except ValueError as e:
    _check("md.concat_clips joins them", False, str(e))


def _kfs_from_record(items):
  return [KF(float(k["t"]), np.array(k["body"], float), {leg: np.array(v, float) for leg, v in k["feet"].items()},
             k.get("ease", "stop")) for k in items]


def part5_regression(runs: Path):
  print(f"\n--- 5. Regression against the benchmark's saved results in {runs}")
  if not runs.is_dir():
    _check("the results folder exists", False, f"no folder {runs}")
    return
  files = sorted(runs.rglob("results.pkl"))  # runs/<experiment>/results.pkl, at any depth
  if not files:
    inside = sorted(p.name for p in runs.iterdir())[:10]
    _check("found results.pkl files", False, f"none under {runs}; it contains {inside or 'nothing'}")
    return
  recs = []
  for f in files:
    with f.open("rb") as fh:  # our own benchmark results (the notebook wrote them)
      recs += [(f.parent.name, r) for r in pickle.load(fh) if r.get("option") == "1_keyframes"]
  usable = [(e, r) for e, r in recs if r.get("keyframes") and r.get("motion") is not None]
  if not _check("found saved keyframe results", bool(usable),
                f"{len(usable)} of {len(recs)} keyframe records in {len(files)} results.pkl have keyframes and a motion"):
    return
  bad_motion, bad_verdict, bad_metrics = [], [], []
  for exp, r in usable:
    tag = f"{exp}/{r['request_id']}/rep{r.get('rep', 0)}"
    interp = (r.get("config") or {}).get("kf_interp", "spline")
    built = build_from_keyframes(_kfs_from_record(r["keyframes"]), interp)
    motion, pre, slow = finalize(built)
    stored = r["motion"]
    if any(np.shape(stored[a]) != np.shape(getattr(motion, a)) for a in ("root_pos", "root_quat", "joint_pos")):
      bad_motion.append(f"{tag}: {len(motion.root_pos)} frames vs {len(stored['root_pos'])} stored")
      continue
    diff = max(float(np.abs(np.asarray(stored[a]) - getattr(motion, a)).max()) for a in ("root_pos", "root_quat", "joint_pos"))
    if diff > 1e-6:
      bad_motion.append(f"{tag}: differs by {diff:.2e}")
    met = evaluate(motion)
    old = r.get("metrics") or {}
    if old.get("physics") != met["physics"]:
      bad_verdict.append(f"{tag}: {met['physics']} vs {old.get('physics')} stored")
    off = {k: (met[k], v) for k, v in old.items() if k in met and v != met[k]
           and not (isinstance(v, float) and isinstance(met[k], float) and math.isclose(v, met[k], abs_tol=1e-6))}
    if off:
      bad_metrics.append(f"{tag}: {dict(list(off.items())[:3])}")
  n = len(usable)
  _check("every rebuilt motion matches the stored one (to 1e-6)", not bad_motion, f"{n - len(bad_motion)}/{n}; " + "; ".join(bad_motion[:3]))
  _check("every physics verdict matches", not bad_verdict, f"{n - len(bad_verdict)}/{n}; " + "; ".join(bad_verdict[:3]))
  _check("every evaluate() number matches", not bad_metrics, f"{n - len(bad_metrics)}/{n}; " + "; ".join(bad_metrics[:3]))
  prompts = [p for p in runs.rglob("keyframes_*.txt") if p.parent.name == "prompts"]
  if not prompts:
    print("note  no saved prompts (runs/<experiment>/prompts/keyframes_*.txt): prompt check skipped")
    return
  latest = max(prompts, key=lambda p: p.name.split("_", 1)[1])
  same = latest.read_text(encoding="utf-8") == keyframe_system()
  _check("keyframe_system() is the benchmark's last keyframe prompt, character for character", same, latest.parent.parent.name)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--runs", type=Path, help="the benchmark's runs/ folder (on Drive) for part 5")
  args = ap.parse_args()
  part1_gate()
  part2_format()
  part3_roundtrip()
  part4_joinable()
  if args.runs:
    part5_regression(args.runs)
  else:
    print("\n--- 5. Regression: skipped (pass --runs <the benchmark's runs/ folder>)")
  print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
  raise SystemExit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
  main()
