"""The Phase 4 test set: motions the tracker never trained on, in one folder.

New in Phase 4. Three parts (general_tracker_plan.md, Phase 4):
  4a  the benchmark's keyframe motions (12 requests x 2 repeats), from its results.pkl,
      with the precision checks of the precise requests (metrics.REQUESTS)
  4b  new Gemini requests written after training (made in Colab; add_new_requests)
  4c  the held-out course tricks, exactly as their specialists were trained
      (tricks/<name>/motion.npz), with the specialist's policy.json to compare against

  <test dir>/motions/<id>.npz    the reference, in the tracker's format
  <test dir>/specialists/<name>.policy.json
  <test dir>/test_manifest.jsonl one line per motion: id, set, source, request, spec, specialist
"""
import json
import pickle
import shutil
from pathlib import Path

from .course import HELD_OUT, TRICKS_DIR
from .metrics import REQUESTS
from .motion import motion_from_dict


class TestSet:
  def __init__(self, root):
    self.root = Path(root)
    self.motions = self.root / "motions"
    self.specialists = self.root / "specialists"
    self.motions.mkdir(parents=True, exist_ok=True)
    self.specialists.mkdir(exist_ok=True)
    self.manifest = self.root / "test_manifest.jsonl"

  def rows(self) -> list[dict]:
    if not self.manifest.exists():
      return []
    return [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

  def ids(self) -> set[str]:
    return {r["id"] for r in self.rows()}

  def add(self, row: dict, motion=None, npz_from=None) -> bool:
    """Add one test motion (an md.Motion, or an existing npz to copy). False if already there."""
    if row["id"] in self.ids():
      return False
    dest = self.motions / f"{row['id']}.npz"
    if motion is not None:
      motion.save_npz(dest)
    else:
      shutil.copyfile(npz_from, dest)
    with self.manifest.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row) + "\n")
    return True


def add_benchmark(ts: TestSet, results_pkl) -> int:
  """4a: every keyframe motion the benchmark saved (its final, gated motion)."""
  specs = {r["id"]: r.get("spec") for r in REQUESTS}
  with open(results_pkl, "rb") as f:  # our own benchmark results (the notebook wrote them)
    recs = [r for r in pickle.load(f) if r.get("option") == "1_keyframes" and r.get("motion") is not None]
  n = 0
  for r in sorted(recs, key=lambda r: (r["request_id"], r.get("rep", 0))):
    row = dict(id=f"bench_{r['request_id']}_r{r.get('rep', 0)}", set="4a", source="benchmark",
               kind=r.get("kind", ""), request=r["request"], request_id=r["request_id"],
               spec=specs.get(r["request_id"]), benchmark_ok=r.get("ok"))
    n += ts.add(row, motion=motion_from_dict(r["motion"]))
  return n


def add_course_held_out(ts: TestSet, tricks_dir: Path = TRICKS_DIR) -> int:
  """4c: the held-out course tricks as their specialists were trained, plus the specialists."""
  n = 0
  for name in HELD_OUT:
    spec_file = ts.specialists / f"{name}.policy.json"
    if not spec_file.exists():
      shutil.copyfile(tricks_dir / name / "policy.json", spec_file)
    row = dict(id=f"course_{name}", set="4c", source="course", kind="course trick", request=name,
               specialist=f"specialists/{name}.policy.json")
    n += ts.add(row, npz_from=tricks_dir / name / "motion.npz")
  return n


def add_new_requests(ts: TestSet, llm, pool_file, train_pool_file, n: int = 30) -> dict:
  """4b: new requests, written now (after training), that aren't like any training
  request, the held-out tricks or the benchmark; each becomes a gated motion."""
  from .pipeline import request_to_motion
  from .request_gen import generate_requests, screen

  train_texts = [json.loads(line)["text"] for line in Path(train_pool_file).read_text(encoding="utf-8").splitlines()]
  requests = generate_requests(llm, pool_file, n_total=round(n * 1.3))
  fresh = [r for r in requests if screen(r["text"], train_texts) is None]
  out = {"requests": len(requests), "not like training": len(fresh), "added": 0, "failed gate": 0}
  for r in fresh:
    if out["added"] >= n:
      break
    rid = f"new_{r['id']}"
    if rid in ts.ids():
      out["added"] += 1
      continue
    rec = request_to_motion(r["text"], llm, request_id=r["id"])
    if not rec["ok"]:
      out["failed gate"] += 1
      continue
    ts.add(dict(id=rid, set="4b", source="gemini_new", kind=r["category"], request=r["text"]),
           motion=motion_from_dict(rec["motion"]))
    out["added"] += 1
  return out
