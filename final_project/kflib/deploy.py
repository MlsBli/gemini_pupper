"""Phase 5: turn the general tracker plus one new motion into a robot trick folder.

New in Phase 5. Nothing here touches the robot: it writes tricks/<name>/ (policy.json,
trick.yaml, motion.npz) in a folder you choose, for the robot's repo. On the robot,
the launch file loads one controller per policy trick at startup (robot/tricks.py),
so a new trick needs a restart: robot/stop.sh, then robot/start.sh. That is the
student's step, at the robot, with the TA's OK (a new trained policy).

  export   pupper-mjlab's own export_trick (unchanged): the tracker's weights with this
           one motion embedded, parity-checked against the trained actor
  checks   the same deploy contract as a course trick policy (keys, network shapes,
           gains, observation layout), the embedded motion is the one we chose, the
           reference never tilts near the robot's tilt e-stop, and robot/tricks.py
           (the robot's own loader) accepts the folder
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REFERENCE_POLICY = REPO / "tricks" / "sneeze" / "policy.json"  # a course trick policy: the contract to match
MAX_BODY_ANGLE_DEG = 60.0  # the course default tilt e-stop while a trick plays (robot/tricks.py); not raised
MAX_REFERENCE_TILT_DEG = 45.0  # refuse references that tilt closer than 15 deg to the e-stop
NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


def export(checkpoint, motion_npz, name: str, out_json, timeout: float = 1800) -> None:
  """pupper-mjlab's export_trick, as a subprocess: weights + one embedded motion."""
  cmd = [sys.executable, "-m", "mjlab.tasks.tracking.config.pupper.export_trick", "--checkpoint", str(checkpoint),
         "--motion-file", str(motion_npz), "--name", name, "--out", str(out_json)]
  p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
  if p.returncode != 0 or not Path(out_json).exists():
    tail = (p.stdout + p.stderr).strip().splitlines()
    raise RuntimeError("export_trick failed: " + (tail[-1] if tail else f"exit {p.returncode}"))
  print((p.stdout.strip().splitlines() or [""])[-1])


def contract_differences(policy: dict, reference: dict) -> list[str]:
  """Everything that differs from a course trick policy, except the motion and its name."""
  out = []
  for key in sorted(set(policy) | set(reference)):
    if key in ("motion", "trick", "layers"):
      continue
    a, b = policy.get(key), reference.get(key)
    if a == b:
      continue
    try:  # numbers (gains, limits, default pose) may differ in the last float digits
      same = np.shape(a) == np.shape(b) and np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-6)
    except (TypeError, ValueError):
      same = False
    if not same:
      out.append(f"{key}: {str(a)[:80]} vs {str(b)[:80]}")
  la, lb = policy.get("layers", []), reference.get("layers", [])
  shape = lambda layers: [(np.shape(x["weights"][0]), x["activation"]) for x in layers]  # noqa: E731
  if shape(la) != shape(lb):
    out.append(f"network layers {shape(la)} vs {shape(lb)}")
  return out


def max_tilt_deg(motion_npz) -> float:
  """The reference body's largest tilt away from upright (roll and pitch, not heading)."""
  with np.load(motion_npz) as z:
    q = z["body_quat_w"][:, 0].astype(float)  # base_link, (w, x, y, z)
  up_z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)  # z of the body's up axis
  return float(np.degrees(np.arccos(np.clip(up_z, -1, 1))).max())


def load_robot_tricks(trick_dir) -> dict:
  """The robot's own loader (robot/tricks.py, no ROS imports) on a folder of tricks."""
  spec = importlib.util.spec_from_file_location("robot_tricks", REPO / "robot" / "tricks.py")
  mod = importlib.util.module_from_spec(spec)
  sys.modules["robot_tricks"] = mod  # its dataclasses look their module up by name
  spec.loader.exec_module(mod)
  return mod.load_tricks(Path(trick_dir))


def install(policy_json, motion_npz, name: str, description: str, out_root, source: dict | None = None) -> dict:
  """Check an exported trick policy and write out_root/<name>/ for the robot. Refuses to
  overwrite. Returns a summary; raises with the reason when a check fails."""
  if not NAME.match(name):
    raise ValueError(f"trick name {name!r}: lowercase letters, digits and _ only (the name on the robot)")
  if not description.strip() or "TODO" in description:
    raise ValueError("write a description: it is what Gemini reads when deciding to play this trick")
  dest = Path(out_root) / name
  if dest.exists():
    raise FileExistsError(f"{dest} exists; pick a new name (trick folders are never overwritten)")
  policy = json.loads(Path(policy_json).read_text())
  if "trick" not in policy or "motion" not in policy:
    raise ValueError("not a trick policy (no trick/motion blocks)")
  diffs = contract_differences(policy, json.loads(REFERENCE_POLICY.read_text()))
  if diffs:
    raise ValueError("differs from a course trick policy: " + "; ".join(diffs))
  with np.load(motion_npz) as z:
    embedded, ours = np.asarray(policy["motion"]["joint_pos"]), z["joint_pos"]
  if embedded.shape != ours.shape or np.abs(embedded - ours).max() > 1e-5:
    raise ValueError(f"the embedded motion {embedded.shape} is not {Path(motion_npz).name} {ours.shape}")
  tilt = max_tilt_deg(motion_npz)
  if tilt > MAX_REFERENCE_TILT_DEG:
    raise ValueError(f"the reference tilts {tilt:.0f} deg, too close to the {MAX_BODY_ANGLE_DEG:.0f} deg tilt e-stop")
  dest.mkdir(parents=True)
  shutil.copyfile(policy_json, dest / "policy.json")
  shutil.copyfile(motion_npz, dest / "motion.npz")
  (dest / "trick.yaml").write_text("\n".join([
    f"# {name}: a motion the general tracker never trained on (final_project, Phase 5).",
    f"# Source: {json.dumps(source or {})}",
    f"description: {json.dumps(description.strip())}",
    "type: policy",
    "policy: policy.json",
    "start_pose: standing",
    "end_pose: standing",
    f"max_body_angle_deg: {MAX_BODY_ANGLE_DEG:g}   # the course default; the reference tilts up to {tilt:.0f} degrees",
  ]) + "\n")
  try:
    loaded = load_robot_tricks(Path(out_root))
  except Exception:
    shutil.rmtree(dest)
    raise
  t = loaded.get(name)
  if t is None:
    shutil.rmtree(dest)
    raise ValueError("robot/tricks.py did not load it")
  return dict(folder=str(dest), duration_s=t.duration, reference_tilt_deg=round(tilt, 1),
              controller=t.controller, description=t.description)


def make_trick(checkpoint, motion_npz, name: str, description: str, out_root, source: dict | None = None) -> dict:
  """Export (pupper-mjlab's export_trick) and install (the checks above) in one go."""
  tmp = Path(out_root) / f".{name}.policy.json"
  tmp.parent.mkdir(parents=True, exist_ok=True)
  try:
    export(checkpoint, motion_npz, name, tmp)
    return install(tmp, motion_npz, name, description, out_root, source)
  finally:
    tmp.unlink(missing_ok=True)
