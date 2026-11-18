# Phase 1 proposal: `final_project/kflib/`, a reusable keyframe library

**Status (2026-11-17): implemented** as `final_project/kflib/` and `final_project/make_motion.py`.
- Colab is the reference environment; pupper-mjlab is pinned to `aa475e9` (`kflib/__init__.py`).
- `run_keyframes` keeps its notebook name; `request_to_motion` is a thin wrapper around it.
- The selftest's parts 1–4 pass locally (18/18). Part 5, the regression against the benchmark's
  `runs/`, needs Drive, so it runs in Colab.

**Goal (from the plan).** Move the CPU half of `pupper_generation_benchmark.ipynb` into plain Python
that Colab (and later a laptop) can import, **with identical behaviour**, plus small tests. Done when:
- the tests pass, and
- a ~10-line script turns one request into a gated `md.Motion` and saves its npz.

## What moves, and where

Only the keyframe path moves. Everything UniMate stays in the notebook: `RigMap`, `unimate_guide`,
`BenchUniMate`, the hybrid and text pipelines, and the worker. The benchmark is finished, so there's
no reason to carry it along.

| module | contents (from notebook cell) | why it's its own file |
|---|---|---|
| `kflib/motion.py` | `quat_to_rpy`, `rpy_series`, `quat_angle_deg`, `make_motion`, `concat_motions`, `stretch_motion`, `feet_series`, `motion_to_dict` / `motion_from_dict` (cell 9) | motion helpers everything else uses |
| `kflib/keyframes.py` | `SCHEMA_LIMITS`, `KF`, `parse_keyframes`, `_quintic`, `keyframe_track`, `build_from_keyframes`, `keyframe_errors` (cell 9); `reach_problems` (cell 13) | the keyframe format and its interpolation + IK |
| `kflib/verify.py` | `PhysicsGate`, `contact_analysis` and its helpers, `slide_stats`, `GATE`, `gate`, `stand_deviation_cm`, `with_stand_ends`, `auto_slow`, `finalize` (cells 9, 13) | the verifier and gate: one place decides pass/fail |
| `kflib/metrics.py` | `evaluate` (cell 9); the precise-request specs and `measure_spec` (cell 12) | scoring, which Phase 4 reuses on tracked motions |
| `kflib/prompt.py` | `_pose_ok`, `probe_limits`, `ROBOT_BRIEF`, `KEYFRAME_FORMAT`, `EXAMPLE_SNEEZE`, `EXAMPLE_STEP`, `KEYFRAME_SYSTEM`, `parse_json` (cell 11) | what Gemini is told; Phase 6 will edit this |
| `kflib/llm.py` | `LLM` (cell 6), `Stopwatch` (cell 13) | backends; latency timing for Phase 6 |
| `kflib/pipeline.py` | `feedback`, `run_keyframes` → renamed `request_to_motion(text, llm)`, returning the same record (cell 13) | request in → gated motion out, up to 3 attempts |

Each module starts with a short docstring saying what it is and which notebook cell it came from, so
you can check the move line by line.

**The only intended changes** (everything else is copied verbatim):
1. **No work at import time.**
   - `PHYS = PhysicsGate()` and `LIM = probe_limits()` become cached functions (`physics_gate()`,
     `reach_limits()`), so `import kflib` is instant. Same objects, same numbers.
   - `KEYFRAME_SYSTEM` is built on first use, because it embeds the probed limits.
2. **No UniMate import.** Cell 9 imports `unimate as U` for `GEN_FPS` and `SKELETON`, but only the
   UniMate parts use them.
3. **`LLM` works outside Colab.**
   - `"colab"` stays as is (free, Colab only).
   - `"aistudio"` reads the key from Colab Secrets in Colab. Elsewhere it reads the `GEMINI_API_KEY`
     environment variable or `~/.config/gemini-pupper/api_key`.
   - **Never a file in the repo.**
4. **Dependency:** pupper-mjlab must be installed (`md`, `get_spec_with_hulls`, the joint constants),
   exactly as in the notebook. Nothing in pupper-mjlab is changed.

## Tests: `kflib/selftest.py`

A plain script that prints PASS/FAIL like the notebook's section 5 and `lab/test.py`; no pytest
needed. Run it with `python -m kflib.selftest`.

1. **The benchmark's self-tests 5a, unchanged:**
   - a still stand passes (HOLDS UP)
   - the course `sit` passes
   - the sneeze example passes
   - both front paws up for 1 s is rejected
   - a too-fast move is rejected
2. **Format errors are readable:** a motion that doesn't end at the stand, keyframes out of order, an
   unknown foot. Each is rejected with the sentence Gemini would see.
3. **Round trip:** `motion_from_dict(motion_to_dict(m))` is identical, and `save_npz` writes a file
   with the 7 arrays the tracker loads, at 50 Hz with 13 bodies.
4. **Joinable** (from Phase 0, question 1): two keyframe motions pass `md.concat_clips`, and their
   root position and heading are continuous at the boundary (≤ 5 mm, ≤ 1°).
5. **Regression against the benchmark** (needs Drive; it's what proves "identical behaviour"). For
   every keyframe record in the benchmark's `runs/`:
   - rebuild from its stored keyframes (`build_from_keyframes` → `finalize`), then
   - check the motion matches the stored one to 1e-6, and the gate verdict and `evaluate()` numbers
     match exactly.

   This needs no LLM calls.

## The 10-line script: `final_project/make_motion.py`

```python
import sys
from kflib.llm import LLM
from kflib.motion import motion_from_dict
from kflib.pipeline import request_to_motion

request = sys.argv[1]                      # e.g. "lift your front left paw for two seconds"
rec = request_to_motion(request, LLM("aistudio"))
print("passed" if rec["ok"] else "failed", rec["reasons"], rec["metrics"]["physics"])
if rec["ok"]:
    motion_from_dict(rec["motion"]).save_npz("motion.npz")   # ready for play_motion / the tracker
```

Running it calls the LLM once to three times. Nothing goes near the robot: it writes a file. The
follow-up check is `play_motion motion.npz` in simulation.

## Order of work (about 1 day)

1. `motion.py`, `keyframes.py`, `verify.py`, then selftest parts 1–4. That runs with no LLM and no
   Drive.
2. `metrics.py`, `prompt.py`, `llm.py`, `pipeline.py`, then the script.
3. One Colab run: selftest including part 5 (the regression), plus the script on two or three
   requests. You run that cell and paste me the output.
4. Your review. After each module I'll offer to walk you through it, so you can explain every line.

**Not in Phase 1:** procedural motions, library assembly and the dataset manifest. Those are Phase 2,
where the source mix and the acceptance rule are marked **(decide)** for you.

## Two questions before I start

1. **Where does `kflib` run?**
   - **(a) Colab only, for now (my recommendation).** pupper-mjlab is already set up there, and a CPU
     runtime is enough for everything in Phase 1.
   - **(b) Also on your Windows laptop.** That means installing pupper-mjlab locally with `uv`. Its GPU
     simulation dependencies may be painful on Windows; an hour, maybe more.

   With (a), Colab gets the code by cloning your GitHub repo (`MlsBli/gemini_pupper`). That means
   committing and pushing `final_project/` (it's untracked today), and I only do that when you say so.
2. **Pin pupper-mjlab to commit `aa475e9`** (the one Phase 0 read, from 2026-11-05) instead of
   `main`? I recommend pinning: the Phase 0 answers and line references stay true, and a course update
   can't change the gate or the tracker under you mid-project. We'd update deliberately if the course
   ships a fix.
