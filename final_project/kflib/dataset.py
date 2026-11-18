"""Dataset folders for the tracker's training motions, and building the library from them.

New in Phase 2. One folder per dataset version (on Drive), never overwritten:

  datasets/<name>/
    motions/<id>.npz   the tracker's motion format (md.Motion.save_npz)
    motions/<id>.json  everything about it: source, group, keyframes, gate report, metrics
    manifest.csv       one row per motion (MANIFEST_FIELDS), appended as motions are added
    requests.jsonl     every generated request and how it went, kept or not
    split.json         the held-out lists and the validation rule, written once

Train / validation split: by *group*, a hash of the group's name (see split_of). A
group is what a motion came from (a request, a procedural seed, a course trick), and
augmented copies share their parent's group, so a motion and its mirror always land
on the same side. The rule needs no list, so the split is fixed from the first motion
and stays fixed as the dataset grows.
"""
import csv
import json
import math
import threading
import zlib
from pathlib import Path

import numpy as np

from mjlab.tasks.tracking.config.pupper import motion_design as md

from .metrics import evaluate
from .motion import make_motion, quat_to_rpy

MANIFEST_FIELDS = [
  "id", "source", "group", "parent", "aug", "kind", "request", "split",
  "duration_s", "verdict", "physics_pos_cm", "physics_tilt_deg",
  "body_height_range_cm", "body_tilt_range_deg", "body_yaw_range_deg", "max_foot_lift_cm",
  "joint_travel_rad", "jerk_rms",
]
VAL_PERCENT = 10  # groups whose hash lands in the first 10 of 100 buckets are validation


def split_of(group: str, val_percent: int = VAL_PERCENT) -> str:
  """'val' or 'train' for a group, from a hash of its name: stable, no list needed."""
  return "val" if zlib.crc32(group.encode()) % 100 < val_percent else "train"


def _json_default(o):
  if isinstance(o, np.generic):
    return o.item()
  if isinstance(o, np.ndarray):
    return o.tolist()
  raise TypeError(f"not JSON serializable: {type(o).__name__}")


class Dataset:
  """A dataset folder. add() writes a motion; nothing is ever overwritten."""

  def __init__(self, root):
    self.root = Path(root)
    self.motions = self.root / "motions"
    self.motions.mkdir(parents=True, exist_ok=True)
    self.manifest = self.root / "manifest.csv"
    self._lock = threading.Lock()  # Gemini motions are generated in parallel threads

  def rows(self) -> list[dict]:
    if not self.manifest.exists():
      return []
    with self.manifest.open(newline="", encoding="utf-8") as f:
      return list(csv.DictReader(f))

  def ids(self) -> set[str]:
    return {r["id"] for r in self.rows()}

  def add(self, motion_id: str, motion: md.Motion, *, source: str, group: str, gate_result: dict,
          parent: str = "", aug: str = "", kind: str = "", request: str = "", extra: dict | None = None) -> dict:
    """Save one gated motion: its npz, its JSON record and a manifest row. Returns the row."""
    if any(c in motion_id for c in '/\\:*?"<>|') or not motion_id:
      raise ValueError(f"bad motion id {motion_id!r}")
    met = evaluate(motion, gate_result["physics"])
    row = dict(id=motion_id, source=source, group=group, parent=parent, aug=aug, kind=kind,
               request=request, split=split_of(group), duration_s=met["duration_s"],
               verdict=met["physics"], physics_pos_cm=met["physics_pos_cm"],
               physics_tilt_deg=met["physics_tilt_deg"], body_height_range_cm=met["body_height_range_cm"],
               body_tilt_range_deg=met["body_tilt_range_deg"], body_yaw_range_deg=met["body_yaw_range_deg"],
               max_foot_lift_cm=met["max_foot_lift_cm"], joint_travel_rad=met["joint_travel_rad"],
               jerk_rms=met["jerk_rms"])
    record = dict(row, gate=dict(ok=gate_result["ok"], reasons=gate_result["reasons"],
                                 physics={k: v for k, v in gate_result["physics"].items() if k != "traj"},
                                 report=gate_result.get("report"), slide=gate_result.get("slide")),
                  metrics=met, **(extra or {}))
    with self._lock:
      npz = self.motions / f"{motion_id}.npz"
      if npz.exists() or motion_id in self.ids():
        raise FileExistsError(f"{motion_id} is already in {self.root}; datasets are never overwritten")
      motion.save_npz(npz)
      (self.motions / f"{motion_id}.json").write_text(json.dumps(record, default=_json_default, indent=1))
      new = not self.manifest.exists()
      with self.manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
          w.writeheader()
        w.writerow(row)
    return row

  def record(self, motion_id: str) -> dict:
    return json.loads((self.motions / f"{motion_id}.json").read_text())

  def load_motion(self, motion_id: str) -> md.Motion:
    """The motion back from its npz (base_link, body 0, is the root)."""
    return motion_from_npz(self.motions / f"{motion_id}.npz")

  def log_request(self, entry: dict) -> None:
    with self._lock, (self.root / "requests.jsonl").open("a", encoding="utf-8") as f:
      f.write(json.dumps(entry, default=_json_default) + "\n")

  def write_split(self, test: dict) -> Path:
    """Write split.json once: the held-out test lists, and the validation rule."""
    path = self.root / "split.json"
    if path.exists():
      raise FileExistsError(f"{path} exists; the split is fixed once written")
    path.write_text(json.dumps(dict(
      validation=f"group hash: zlib.crc32(group) % 100 < {VAL_PERCENT} (kflib.dataset.split_of)",
      test=test), indent=1))
    return path


def motion_from_npz(path) -> md.Motion:
  """An md.Motion from a tracker npz (root = body 0, base_link)."""
  with np.load(path) as z:
    if str(z["body_names"][0]) != "base_link":
      raise ValueError(f"{path}: body 0 is {z['body_names'][0]}, expected base_link")
    return make_motion(z["body_pos_w"][:, 0].astype(float), z["body_quat_w"][:, 0].astype(float),
                       z["joint_pos"].astype(float))


REPORT_FIELDS = ("duration_s", "body_height_range_cm", "body_tilt_range_deg", "body_yaw_range_deg",
                 "max_foot_lift_cm", "joint_travel_rad")


def report(ds: Dataset, png=None) -> dict:
  """Counts by source, split and verdict, and (with png) histograms of the motion ranges
  by source: the Phase 2 "is it broad, not one kind of motion?" check."""
  from collections import Counter

  rows = ds.rows()
  summary = dict(motions=len(rows),
                 seconds=round(sum(float(r["duration_s"]) for r in rows), 1),
                 by_source=dict(Counter(r["source"] for r in rows)),
                 by_split=dict(Counter(r["split"] for r in rows)),
                 by_verdict=dict(Counter(r["verdict"] for r in rows)),
                 by_aug=dict(Counter(r["aug"].split()[0] if r["aug"] else "original" for r in rows)))
  if png and rows:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted({r["source"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, field in zip(axes.flat, REPORT_FIELDS):
      data = [[float(r[field]) for r in rows if r["source"] == s] for s in sources]
      ax.hist(data, bins=20, stacked=True, label=sources)
      ax.set_title(field)
    axes.flat[0].legend()
    fig.suptitle(f"{ds.root.name}: {summary['motions']} motions, {summary['seconds'] / 60:.0f} min")
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)
  return summary


# ---------------------------------------------------------------------------- the library
def _quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b.T
  return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def place(motion: md.Motion, x: float, y: float, yaw_deg: float) -> md.Motion:
  """The same motion, turned and shifted on the ground so it starts at (x, y) facing yaw_deg.

  Only the root's horizontal position and heading change (a rigid move in the ground
  plane); heights, tilts and joint angles stay exactly the same.
  """
  x0, y0 = motion.root_pos[0][:2]
  d = math.radians(yaw_deg - quat_to_rpy(motion.root_quat[0])[2])
  c, s = math.cos(d), math.sin(d)
  rel = motion.root_pos[:, :2] - [x0, y0]
  pos = motion.root_pos.copy()
  pos[:, 0] = x + c * rel[:, 0] - s * rel[:, 1]
  pos[:, 1] = y + s * rel[:, 0] + c * rel[:, 1]
  quat = _quat_mul(np.array([math.cos(d / 2), 0.0, 0.0, math.sin(d / 2)]), motion.root_quat)
  return make_motion(pos, quat, motion.joint_pos)


def build_library(motions: dict[str, md.Motion], path=None, tol_z_m: float = 0.005,
                  tol_tilt_deg: float = 1.0) -> md.Motion:
  """Join motions (name -> motion, in that order) into one training library.

  Each motion is placed (see place) where the previous one ended, so the reference
  never jumps on the ground at a seam (the Phase 0 caveat: md.concat_clips checks only
  the joints). Height and tilt can't be fixed that way, so those are checked: every
  seam must match within tol_z_m and tol_tilt_deg, or this raises. md.concat_clips then
  checks the joints (within 0.05 rad) and writes the clip table. With `path`, the
  library is saved (save_npz once, so velocities at the seams come out right).
  """
  clips, prev = [], None
  for name, m in motions.items():
    if prev is not None:
      end_rpy = quat_to_rpy(prev.root_quat[-1])
      m = place(m, prev.root_pos[-1][0], prev.root_pos[-1][1], end_rpy[2])
      dz = abs(m.root_pos[0][2] - prev.root_pos[-1][2])
      dtilt = np.abs(quat_to_rpy(m.root_quat[0])[:2] - end_rpy[:2]).max()
      if dz > tol_z_m or dtilt > tol_tilt_deg:
        raise ValueError(f"seam before {name}: height differs by {1000 * dz:.1f} mm, tilt by {dtilt:.1f} deg")
    clips.append(md.Clip(name, m))
    prev = m
  library = md.concat_clips(clips)
  if path is not None:
    library.save_npz(path)
  return library
