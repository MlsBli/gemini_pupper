"""Fill a dataset folder with gated motions: procedural, course tricks, Gemini requests.

New in Phase 2. Every source goes through the same steps: keyframes -> parse ->
build (IK) -> reach check -> gate. Only passing motions are added (decided: gate
pass, plain-replay FALLS rejected). Each passing motion is also added mirrored, and
a third of them time-scaled (x0.8-1.25), with the same group as their parent. Ids
already in the dataset are skipped, so a run can be stopped and started again.

    python -m kflib.generate procedural --dataset <dir> --n 240 [--workers 4]
    python -m kflib.generate course --dataset <dir> [--tricks pee sit ...]
The Gemini source needs an LLM; in Colab: add_gemini(ds, requests, LLM("colab")).
"""
import argparse
import concurrent.futures as cf
import zlib

from . import augment
from .dataset import Dataset
from .keyframes import build_from_keyframes, parse_keyframes, reach_problems
from .procedural import ProcConfig, random_keyframes
from .verify import gate

TIME_SCALE = (0.8, 1.25)


def gate_keyframes(kfs) -> tuple[object, dict | None, list[str]]:
  """(motion, gate result, problems) for parsed keyframes; problems empty = keep it."""
  motion = build_from_keyframes(kfs)
  errs = reach_problems(motion, kfs)
  g = gate(motion)
  return motion, g, errs + g["reasons"]


def variants(base_id: str, kfs) -> list[tuple[str, str, list]]:
  """The augmented copies to try: (id, aug, keyframes). Mirror always; a time scale
  for a third of the motions (chosen by a hash of the id, so it's repeatable)."""
  out = [(f"{base_id}_m", "mirror", augment.mirror(kfs))]
  h = zlib.crc32(base_id.encode())
  if h % 3 == 0:
    f = round(TIME_SCALE[0] + (h % 1000) / 999 * (TIME_SCALE[1] - TIME_SCALE[0]), 3)
    out.append((f"{base_id}_t{f:g}", f"time x{f:g}", augment.time_scale(kfs, f)))
  return out


def _try(motion_id, kfs):
  """Worker: gate one keyframe list; returns what add() needs (picklable)."""
  kfs, errs = augment.check(kfs)
  if errs:
    return motion_id, None, None, errs
  motion, g, errs = gate_keyframes(kfs)
  g = dict(g, physics={k: v for k, v in g["physics"].items() if k != "traj"} | {"traj": None})
  return motion_id, motion, g, errs


def add_keyframe_motion(ds: Dataset, base_id: str, kfs, *, source: str, group: str, kind: str,
                        request: str = "", extra: dict | None = None, pool=None) -> dict:
  """Gate a keyframe motion, then (only if it passed) its variants; add the ones that pass.
  Returns counts. Variants are gated in parallel when a process pool is given."""
  counts = {"added": 0, "rejected": 0}
  have = ds.ids()

  def keep(mid, aug, k, motion, g, errs):
    if errs:
      counts["rejected"] += 1
      return False
    ds.add(mid, motion, source=source, group=group, gate_result=g, parent=base_id if aug else "",
           aug=aug, kind=kind, request=request, extra=dict(keyframes=augment.to_items(k), **(extra or {})))
    counts["added"] += 1
    return True

  if base_id not in have and not keep(base_id, "", kfs, *_try(base_id, kfs)[1:]):
    return counts  # variants only of a motion that passed itself
  jobs = [j for j in variants(base_id, kfs) if j[0] not in have]
  run = pool.map if pool else map
  for (mid, aug, k), (_, motion, g, errs) in zip(jobs, run(_try, [j[0] for j in jobs], [j[2] for j in jobs])):
    keep(mid, aug, k, motion, g, errs)
  return counts


def add_procedural(ds: Dataset, seeds, cfg: ProcConfig = ProcConfig(), workers: int = 1,
                   chunk: int = 32) -> dict:
  """Procedural motions for these seeds (id proc_<seed>), with their variants.

  Seeds go in chunks. In each chunk, every original motion is gated in parallel
  (`workers` processes); then the variants of the ones that passed, also in
  parallel. Results are added in seed order, so the dataset comes out the same
  whatever `workers` is, and a stopped run loses at most one chunk.
  """
  total = {"added": 0, "rejected": 0, "format": 0}
  seeds = list(seeds)
  pool = cf.ProcessPoolExecutor(workers) if workers > 1 else None
  run = (lambda f, *its: pool.map(f, *its, chunksize=1)) if pool else map
  try:
    for start in range(0, len(seeds), chunk):
      have = ds.ids()
      bases = []  # (id, seed, keyframes)
      for seed in seeds[start:start + chunk]:
        kfs, errs = parse_keyframes(random_keyframes(seed, cfg))
        if errs:
          total["format"] += 1
        else:
          bases.append((f"proc_{seed:05d}", seed, kfs))

      def keep(mid, seed, aug, k, motion, g, errs):
        if errs:
          total["rejected"] += 1
          return False
        ds.add(mid, motion, source="procedural", group=f"proc_{seed:05d}", gate_result=g,
               parent=f"proc_{seed:05d}" if aug else "", aug=aug, kind="procedural",
               extra=dict(keyframes=augment.to_items(k), seed=seed))
        total["added"] += 1
        return True

      todo = [b for b in bases if b[0] not in have]
      passed = {b[0] for b in bases if b[0] in have}
      for (mid, seed, kfs), (_, motion, g, errs) in zip(todo, run(_try, [b[0] for b in todo], [b[2] for b in todo])):
        if keep(mid, seed, "", kfs, motion, g, errs):
          passed.add(mid)
      jobs = [(vid, seed, aug, vk) for mid, seed, kfs in bases if mid in passed
              for vid, aug, vk in variants(mid, kfs) if vid not in have]
      for (vid, seed, aug, vk), (_, motion, g, errs) in zip(jobs, run(_try, [j[0] for j in jobs], [j[3] for j in jobs])):
        keep(vid, seed, aug, vk, motion, g, errs)
      print(f"seeds {seeds[start]}-{seeds[min(start + chunk, len(seeds)) - 1]}: total {total}", flush=True)
  finally:
    if pool:
      pool.shutdown()
  return total


def add_course(ds: Dataset, names) -> dict:
  """Course tricks (course.py) that pass the gate. They have no keyframes, so no variants."""
  from .course import course_motion

  out = {}
  for name in names:
    m = course_motion(name)
    g = gate(m)
    out[name] = "added" if g["ok"] else f"rejected: {g['reasons'][:1]}"
    if g["ok"] and f"course_{name}" not in ds.ids():
      ds.add(f"course_{name}", m, source="course", group=f"course_{name}", gate_result=g, kind="course",
             extra=dict(trick=name))
  return out


class _LLMOutsideLock:
  """Calls the LLM with the simulator lock released, so other threads can build and
  gate motions while this one waits for its reply (~16 s)."""

  def __init__(self, llm, lock):
    self.llm, self.lock = llm, lock
    self.backend, self.model = getattr(llm, "backend", ""), getattr(llm, "model", "")

  def __call__(self, system, user):
    self.lock.release()
    try:
      return self.llm(system, user)
    finally:
      self.lock.acquire()


def add_gemini(ds: Dataset, requests: list[dict], llm, threads: int = 1) -> dict:
  """Gemini keyframe motions for requests ({"id", "text", "category"}), with their variants.

  Every request and its attempts go to requests.jsonl, kept or not. With threads > 1,
  LLM calls overlap, but only one thread at a time builds or gates a motion: the IK
  (KIN) and the physics gate each reuse one MuJoCo state, which isn't thread-safe.
  (Latency numbers in the log then include waiting for that lock.)
  """
  import threading

  from .pipeline import request_to_motion

  sim_lock = threading.Lock()

  def one(req):
    with sim_lock:
      return _one(req)

  def _one(req):
    rid = f"gem_{req['id']}"
    if rid in ds.ids():
      return "skipped"
    rec = request_to_motion(req["text"], _LLMOutsideLock(llm, sim_lock), request_id=req["id"])
    ds.log_request(dict(id=req["id"], text=req["text"], category=req.get("category", ""), ok=rec["ok"],
                        reasons=rec["reasons"], attempts=rec.get("attempts"), total_s=rec.get("total_s")))
    if not rec["ok"] or not rec.get("keyframes"):
      return "failed"
    kfs, errs = augment.check(augment.from_record(rec["keyframes"]))
    if errs:
      return "failed"
    c = add_keyframe_motion(ds, rid, kfs, source="gemini", group=rid, kind=req.get("category", ""),
                            request=req["text"], extra=dict(attempts=rec.get("attempts")))
    return f"added {c['added']}"

  results = {}
  with cf.ThreadPoolExecutor(threads) as ex:
    for req, res in zip(requests, ex.map(one, requests)):
      results[req["id"]] = res
      print(f"{req['id']}: {res}", flush=True)
  return results


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("source", choices=["procedural", "course"])
  ap.add_argument("--dataset", required=True)
  ap.add_argument("--n", type=int, default=240, help="procedural: how many seeds to try")
  ap.add_argument("--start", type=int, default=0, help="procedural: first seed")
  ap.add_argument("--workers", type=int, default=1, help="processes (use your CPU core count)")
  ap.add_argument("--chunk", type=int, default=32, help="procedural: seeds per chunk (progress and resume unit)")
  ap.add_argument("--tricks", nargs="*", help="course: which tricks (default: the training ones)")
  args = ap.parse_args()
  ds = Dataset(args.dataset)
  if args.source == "procedural":
    print(add_procedural(ds, range(args.start, args.start + args.n), workers=args.workers, chunk=args.chunk))
  else:
    from .course import TRAIN

    print(add_course(ds, args.tricks or TRAIN))


if __name__ == "__main__":
  main()
