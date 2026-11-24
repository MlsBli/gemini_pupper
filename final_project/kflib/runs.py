"""Bookkeeping for training runs: the library order, the fixed evaluation set,
finding checkpoints across Colab sessions, and summarizing scores.

New in Phase 3 (v1). Training writes checkpoints to <runs>/pupper_tracking/
<date>_<run_name>/model_<iteration>.pt; a resumed session starts a new folder and
keeps counting iterations, so a run's checkpoints are spread over several folders.
"""
import re
import statistics
from pathlib import Path

import numpy as np

from .dataset import Dataset


def library_ids(ds: Dataset, split: str = "train", seed: int = 0) -> list[str]:
  """Every motion of a split (augmented copies included), in a fixed shuffled order.

  Shuffled so the seams (where one motion hands over to the next during training)
  join many different pairs, rather than a motion and its own mirror every time.
  """
  ids = sorted(r["id"] for r in ds.rows() if r["split"] == split)
  np.random.default_rng(seed).shuffle(ids)
  return ids


def eval_set(ds: Dataset, n_train: int = 30, seed: int = 0) -> list[dict]:
  """The fixed set every checkpoint is scored on: all validation originals (never
  trained on, not even their mirrors), plus n_train training originals spread over
  the sources (is it learning what it was shown?)."""
  orig = [r for r in ds.rows() if not r["aug"]]
  val = sorted((r for r in orig if r["split"] == "val"), key=lambda r: r["id"])
  train = sorted((r for r in orig if r["split"] == "train"), key=lambda r: r["id"])
  rng = np.random.default_rng(seed)
  sources = sorted({r["source"] for r in train})
  per = {s: [r for r in train if r["source"] == s] for s in sources}
  pick = []
  for s in sources:  # in proportion to the source, at least one each
    k = max(1, round(n_train * len(per[s]) / len(train)))
    pick += [per[s][i] for i in sorted(rng.choice(len(per[s]), size=min(k, len(per[s])), replace=False))]
  return val + pick[:n_train]


_CKPT = re.compile(r"model_(\d+)\.pt$")


def checkpoints(runs_root, run_name: str) -> list[tuple[int, Path]]:
  """All (iteration, path) of a run, over every session's folder, oldest first."""
  base = Path(runs_root) / "pupper_tracking"
  found = []
  for d in base.glob(f"*_{run_name}"):
    if d.is_dir():
      for f in d.iterdir():
        m = _CKPT.search(f.name)
        if m:
          found.append((int(m.group(1)), f))
  return sorted(found)


def latest_checkpoint(runs_root, run_name: str) -> tuple[int, Path | None]:
  """(iteration, path) of the newest checkpoint of a run, or (0, None) if there is none."""
  c = checkpoints(runs_root, run_name)
  return c[-1] if c else (0, None)


def chunk_plan(latest: int, resumed: bool, target: int, chunk: int) -> tuple[int, int]:
  """(iterations to ask learn() for, index of the last iteration it will run).

  rsl_rl counts iterations from 0 and names a checkpoint after the last index it
  ran: 5,000 iterations from scratch end in model_4999.pt. A resume starts again at
  that index (it reruns index 4999), so to end exactly at the next round number the
  resumed chunk asks for one more. Returns (0, latest) when the target is reached.
  """
  if not resumed:
    n = min(chunk, target)
    return n, n - 1
  if latest >= target - 1:
    return 0, latest
  end = min(latest + chunk, target - 1)
  return end - latest + 1, end


def summarize(rows: list[dict]) -> dict:
  """Per split: how many motions the policy holds up / wobbles / drops, the median and
  worst errors, and the worst five motions (the ones to watch)."""
  out = {}
  for split in sorted({r["split"] for r in rows}):
    rs = [r for r in rows if r["split"] == split and r["mode"] == "policy" and r["verdict"]]
    if not rs:
      continue
    pos = [float(r["pos_cm"]) for r in rs]
    tilt = [float(r["tilt_deg"]) for r in rs]
    worst = sorted(rs, key=lambda r: -(float(r["pos_cm"]) / 3 + float(r["tilt_deg"]) / 10))[:5]
    out[split] = dict(
      n=len(rs), holds_up=sum(r["verdict"] == "HOLDS UP" for r in rs), wobbles=sum(r["verdict"] == "WOBBLES" for r in rs),
      falls=sum(r["verdict"] == "FALLS" for r in rs), median_cm=round(statistics.median(pos), 2),
      median_deg=round(statistics.median(tilt), 1), worst_cm=max(pos), worst_deg=max(tilt),
      worst=[f"{r['id']} ({r['source']}): {r['verdict']} {r['pos_cm']} cm {r['tilt_deg']} deg" for r in worst],
    )
  return out
