"""Phase 4 results: the zero-shot table, precision on the tracked motions, and a figure.

New in Phase 4. Input: the rows run_many (track.py) returns, one per motion and
driver ("plain", "v1", "specialist"), each with the reference's test set (4a/4b/4c).
"""
import statistics
from pathlib import Path

from .metrics import SPECS
from .track import load_tracked

SET_NAMES = {"4a": "benchmark requests", "4b": "new requests", "4c": "held-out course tricks"}


def table(rows: list[dict]) -> list[dict]:
  """Per test set and driver: how many motions stay up (not FALLS), how many HOLD UP,
  and the median worst errors."""
  out = []
  for s in sorted({r["set"] for r in rows}):
    for label in ("plain", "v1", "specialist"):
      rs = [r for r in rows if r["set"] == s and r["label"] == label and r.get("verdict")]
      if not rs:
        continue
      out.append(dict(
        set=s, name=SET_NAMES.get(s, s), driver=label, n=len(rs),
        stays_up=sum(r["verdict"] != "FALLS" for r in rs), holds_up=sum(r["verdict"] == "HOLDS UP" for r in rs),
        median_cm=round(statistics.median(r["pos_cm"] for r in rs), 1),
        median_deg=round(statistics.median(r["tilt_deg"] for r in rs), 1),
        median_joint_rad=round(statistics.median(r["joint_rms_rad"] for r in rs), 3),
      ))
  return out


def precision(rows: list[dict], ts, results_dir, label: str = "v1") -> list[dict]:
  """For the benchmark's precise requests: the check on the reference, and the same check
  on what the robot actually did under `label` (does tracking keep the numbers?).
  ts: the TestSet (testset.py); results_dir: where run_many wrote the results."""
  from .dataset import motion_from_npz

  by_id = {r["id"]: r for r in ts.rows()}
  out = []
  for r in rows:
    t = by_id.get(r["id"], {})
    if r["label"] != label or not t.get("spec") or not r.get("verdict"):
      continue
    name, kw = t["spec"]
    ref = SPECS[name](motion_from_npz(ts.motions / f"{r['id']}.npz"), **kw)
    got = SPECS[name](load_tracked(Path(results_dir) / f"{r['id']}__{label}.json"), **kw)
    out.append(dict(id=r["id"], check=name, reference_ok=ref["ok"], reference=ref["measured"],
                    tracked_ok=got["ok"], tracked=got["measured"]))
  return out


def figure(rows: list[dict], png) -> None:
  """Left: share of motions that HOLD UP, per test set and driver. Right: each motion's
  worst tilt, plain replay vs v1 (below the diagonal = v1 tracks it better)."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  t = table(rows)
  sets = sorted({r["set"] for r in t})
  drivers = [d for d in ("plain", "v1", "specialist") if any(r["driver"] == d for r in t)]
  fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
  w = 0.8 / len(drivers)
  for i, d in enumerate(drivers):
    xs, ys, ns = [], [], []
    for j, s in enumerate(sets):
      r = next((r for r in t if r["set"] == s and r["driver"] == d), None)
      if r:
        xs.append(j + i * w)
        ys.append(100 * r["holds_up"] / r["n"])
        ns.append(r["n"])
    bars = a.bar(xs, ys, w, label=d)
    for bar, n in zip(bars, ns):
      a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"n={n}", ha="center", fontsize=8)
  a.set_xticks([j + w * (len(drivers) - 1) / 2 for j in range(len(sets))])
  a.set_xticklabels([f"{s}\n{SET_NAMES.get(s, s)}" for s in sets])
  a.set_ylabel("% of motions that HOLD UP")
  a.set_ylim(0, 110)
  a.legend()
  plain = {r["id"]: r for r in rows if r["label"] == "plain" and r.get("verdict")}
  all_pts = []
  for s in sets:
    pts = [(plain[r["id"]]["tilt_deg"], r["tilt_deg"]) for r in rows
           if r["label"] == "v1" and r["set"] == s and r.get("verdict") and r["id"] in plain]
    if pts:
      b.scatter(*zip(*pts), label=SET_NAMES.get(s, s), alpha=0.75)
      all_pts += pts
  lim = max([max(p) for p in all_pts] + [10]) * 1.1
  b.set_xlim(0, lim)
  b.set_ylim(0, lim)
  b.plot([0, lim], [0, lim], "k--", lw=0.8)
  b.axhline(10, color="grey", lw=0.6, ls=":")
  b.set_xlabel("worst tilt, plain replay (deg)")
  b.set_ylabel("worst tilt, v1 zero-shot (deg)")
  b.set_xscale("symlog", linthresh=10)
  b.set_yscale("symlog", linthresh=10)
  b.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(png, dpi=100)
  plt.close(fig)
