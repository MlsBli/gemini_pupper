"""Score motions with pupper-mjlab's play_motion: plain replay, or a trained tracking policy.

New in Phase 3. play_motion is the course's scoring tool (the same physics as
training, the nominal robot, terminations off). It prints the worst body position
and orientation error over the motion and a verdict:
  HOLDS UP   position error <= 3 cm and tilt <= 10 deg throughout
  WOBBLES    beyond that, but never past 10 cm / 45 deg
  FALLS      past 10 cm or 45 deg at some point
This module runs it as a subprocess, unchanged, a few motions at a time, and reads
those lines back. Nothing here moves the robot.
"""
import concurrent.futures as cf
import csv
import re
import subprocess
import sys
from pathlib import Path

PLAY_MOTION = "mjlab.tasks.tracking.config.pupper.play_motion"
_POS = re.compile(r"worst body position error\s+([\d.]+) cm")
_TILT = re.compile(r"worst body orientation error\s+([\d.]+) deg")
_VERDICT = re.compile(r"verdict\s+(HOLDS UP|WOBBLES|FALLS)(?: (?:from|at) t = ([\d.]+) s)?")


def parse(output: str) -> dict:
  """The numbers play_motion printed (verdict None if it didn't finish)."""
  pos, tilt, verdict = _POS.search(output), _TILT.search(output), _VERDICT.search(output)
  return dict(pos_cm=float(pos.group(1)) if pos else None, tilt_deg=float(tilt.group(1)) if tilt else None,
              verdict=verdict.group(1) if verdict else None,
              at_s=float(verdict.group(2)) if verdict and verdict.group(2) else None)


def play_motion(npz, checkpoint=None, timeout: float = 900) -> dict:
  """Run play_motion on one motion npz; with a checkpoint, the policy plays it."""
  cmd = [sys.executable, "-m", PLAY_MOTION, str(npz)]
  if checkpoint:
    cmd += ["--checkpoint", str(checkpoint)]
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
  except subprocess.TimeoutExpired:
    return dict(pos_cm=None, tilt_deg=None, verdict=None, at_s=None, error=f"timed out after {timeout:.0f} s")
  r = parse(out)
  if r["verdict"] is None:
    r["error"] = out.strip().splitlines()[-1][:200] if out.strip() else f"exit {p.returncode}, no output"
  return r


SCORE_FIELDS = ["id", "group", "source", "kind", "split", "mode", "verdict", "pos_cm", "tilt_deg", "at_s", "error"]


def score(motions: list[dict], checkpoint=None, out_csv=None, workers: int = 4, plain: bool = True) -> list[dict]:
  """Score each motion ({"id", "npz", and any of group/source/kind/split}) by plain replay
  (if `plain`) and by the checkpoint (if given). Rows are appended to out_csv as they finish."""
  jobs = [(m, None) for m in motions if plain] + ([(m, checkpoint) for m in motions] if checkpoint else [])
  rows = []
  out = Path(out_csv) if out_csv else None
  if out and not out.exists():
    with out.open("w", newline="") as f:
      csv.DictWriter(f, fieldnames=SCORE_FIELDS).writeheader()
  with cf.ThreadPoolExecutor(workers) as ex:
    futures = {ex.submit(play_motion, m["npz"], ck): (m, ck) for m, ck in jobs}
    for fut in cf.as_completed(futures):
      m, ck = futures[fut]
      r = fut.result()
      row = {k: m.get(k, "") for k in ("id", "group", "source", "kind", "split")}
      row.update(mode="policy" if ck else "plain", error="", **r)
      rows.append(row)
      print(f"{row['id']:28} {row['mode']:6} {str(row['verdict']):9} {row['pos_cm']} cm {row['tilt_deg']} deg"
            + (f"  ({row['error']})" if row["error"] else ""), flush=True)
      if out:
        with out.open("a", newline="") as f:
          csv.DictWriter(f, fieldnames=SCORE_FIELDS).writerow(row)
  return rows


def compare(rows: list[dict]) -> list[dict]:
  """One line per motion: plain replay next to the policy."""
  by = {}
  for r in rows:
    by.setdefault(r["id"], {"id": r["id"], "split": r["split"], "source": r["source"]})[r["mode"]] = r
  out = []
  for d in by.values():
    p, q = d.get("plain", {}), d.get("policy", {})
    out.append(dict(id=d["id"], split=d["split"], source=d["source"],
                    plain=f"{p.get('verdict')} {p.get('pos_cm')} cm {p.get('tilt_deg')} deg" if p else "",
                    policy=f"{q.get('verdict')} {q.get('pos_cm')} cm {q.get('tilt_deg')} deg" if q else "",
                    tracks=q.get("verdict") == "HOLDS UP"))
  return sorted(out, key=lambda d: (d["split"], d["id"]))
