# Phase 2 proposal: generating the training motions

**Status (2026-11-17): decided; in progress.** The student picked all four recommendations:
- **Mix:** ~250 Gemini + ~240 procedural + the course tricks not held out. Then mirror everything and
  time-scale a third.
- **Keep:** what the gate passes (FALLS rejected).
- **Hold out:** `sneeze`, `push_up` and `stand_sit_shake_sit_stand`.
- **LLM:** Colab AI, with a 20-request pilot first.

**Progress (2026-11-17):**
- **Built:**
  - `kflib/augment.py`: mirror, time scale, jitter, all on keyframes.
  - `kflib/procedural.py`: random keyframe lists.
  - `kflib/dataset.py`: dataset folder, manifest, group-hash split, `place`, `build_library` with
    the seam check, `report`.
  - `kflib/course.py`: course tricks.
  - `kflib/generate.py`: the procedural, course and Gemini sources. Gemini runs in parallel
    threads; only one thread at a time uses the simulator.
- **Selftest:** 31/31, with a new part 6 for these pieces.
- **Procedural motions:** at 75% of the measured reach, 31/40 pass the gate (19 hold up, 12 wobble).
  Mirrored copies pass 31/31; time-scaled copies pass ~97%.
- **Pilot:** a local dataset of 46 motions.
- **Speed:** the procedural step now gates original motions in parallel too: ~4 s per seed on the
  laptop's 8 cores, so 240 seeds take ~17 min instead of ~70. The output is identical to the serial
  version (checked on 16 seeds: same ids, verdicts and splits).
- **Bug found and fixed:** the IK and the physics gate each reuse one MuJoCo state. Two Gemini threads
  gating at once crashed MuJoCo; a lock now lets only one thread at a time touch the simulator.

- **Request generator** (`kflib/request_gen.py`): categories in the decided proportions. It drops
  the held-out tricks, near-copies of the benchmark's 12 requests, and duplicates. Requests that
  differ only in their numbers ("paw up 3 cm" vs "5 cm") are kept, since they're different motions.
- **Colab notebook** (`phase2_generate.ipynb`): setup → selftest → split.json (written once) →
  requests → 20-request pilot → full Gemini run → procedural + `sit` → report → a 10-motion library
  for Phase 3. Its logic was run locally with a stand-in LLM; the selftest is 40/40.

- **Procedural set generated (2026-11-18):** `C:/Users/M/pupper_datasets/v1`, 93 MB, made in 11 min on 8 cores.
  - 409 procedural motions (177 of 240 seeds passed, 74%) plus `sit`: 410 motions, 40 min of motion.
  - Split: 372 train, 38 val. `split.json` is written.
  - Verdicts: 67% HOLDS UP, 33% WOBBLES (the benchmark's Gemini motions held up 54% of the time).
  - **Skew:** 78% of motions lift a paw. Big height changes are rare: the median height range is 1 cm,
    the lean before a lift.
- **Rebalanced and regenerated (2026-11-18, the student's call):** the paw-lift chance per move is
  now 0.15 (was 0.3), and pose moves pick height more often (weights 0.3 for dz, 0.2 pitch, 0.15
  roll, 0.15 yaw, 0.1 x, 0.1 y). The new `v1`:
  - 391 motions, 33 min, 77 MB; 170/240 seeds pass.
  - Paw lifts: 78% → **46%**. Height range of 2 cm or more: 32% → **52%** (median 1.0 → 2.3 cm).
    Median tilt: 12° → 14°.
  - Trade-off: these hold up more often, 81% against 67%: paw lifts were what made motions wobble.
    The Gemini half (54% held up in the benchmark) brings harder motions, so judge the combined set.
  - The old set is kept as `v1_lift_heavy`.
  - Each record now stores its `proc_config`.

**Two findings that change the decisions above:**
1. **Most recorded course tricks fail our gate.** Of the 7, only `sit` and `sneeze` pass. The
   recordings have joints at their limits (push_up, stand_sit_stand, stand_sit_shake_sit_stand), feet
   sliding 8 cm (stand_downward_dog), and a too-fast blend at the end (pee). So "train on the other
   four" would mean training on `sit` alone, unless the gate is relaxed for them.
   **Decided 2026-11-17: train on `sit` only.**
2. **Leak risk for 4c.** The keyframe prompt's worked sneeze example *is* the held-out course sneeze.
   Generated requests must therefore avoid the held-out tricks (sneeze, push-ups, sit/shake), or the
   zero-shot test on them isn't zero-shot.

**Goal (from the plan).** About 500 gated motions with a manifest and a fixed held-out split, and a
broad spread of motion kinds. Every motion starts and ends at the stand (x = y = 0, yaw = 0), so any
two of them can be joined into the training library (Phase 0, question 1).

## What we have to work with (measured)

- **The gate takes about 1.5 s per motion** on one laptop core: build + IK ≈ 0.85 s and the gate ≈ 0.6 s
  for a 2.5 s motion. So 500 motions are ~15 min of CPU per pass, and procedural motions with rejection
  sampling are a coffee break, not a day. Colab's extra cores make it faster still.
- **The LLM call is ~16 s** (the benchmark's median). The benchmark's keyframe motions passed
  100% within 3 attempts. So 250 Gemini motions are ~1.5–2 h unattended, less with parallel requests.
- **Course tricks:** the repo's `tricks/` has 7 that start *and* end standing:
  `pee`, `push_up`, `sneeze`, `stand_downward_dog`, `stand_sit_shake_sit_stand`, `stand_sit_stand`,
  plus pupper-mjlab's `sit.npz`. They're long (7–15 s), so 7 tricks are only ~4% of the library by
  time. The recorded ones aren't perfectly closed:
  - they end 1–4 cm away from where they started;
  - their stand is 0.5–2 cm taller than ours (0.166–0.178 m vs 0.159 m).

  So they'd need blending into our stand at both ends, plus lining up at the join. The other 8 end
  lying down and can't join a stand-to-stand library.
- Every specialist trick has a trained `policy.json`, so whichever course tricks we hold out become
  Phase 4c's comparison.

## The sources, and how each one is made

1. **Gemini keyframes from generated requests.**
   - An LLM first writes requests by category: precise (heights, angles, a specific paw, counts,
     holds), moods and styles, dog actions, sequences, body-pose sweeps, small hops.
   - Duplicates are dropped, and so is anything matching the benchmark's 12 requests: those are
     Phase 4a's test motions.
   - Each request then goes through `request_to_motion` (up to 3 attempts, exactly as in the
     benchmark).
   - Every request and every attempt is logged.
2. **Procedural random keyframes**, in new `kflib/procedural.py`. Random but sensible keyframe lists
   between two stands, seeded per motion so any motion can be rebuilt:
   - **Moves (2–6 of them):** body poses using 1–3 of dz / pitch / roll / yaw / x / y, inside a
     fraction of the probed reach. Also paw lifts that follow the lean-away rule (shift the body
     ~2 cm away first, lift 2–6 cm), and short "flow" wiggle series.
   - **Timing:** 0.3–1.0 s per move, random stop/flow, random holds.
   - **Filter:** the same gate decides what's kept. I'll report the acceptance rate.
3. **Course tricks** (the 7 above). Each is blended into our stand at both ends (`with_stand_ends`)
   and gated like everything else.
4. **Augmentation**, in new `kflib/augment.py`, done **on the keyframes**, before IK:
   - **Mirror:** flip y, roll and yaw, and swap left and right feet.
   - **Time scale:** ×0.8–1.25.
   - **Jitter:** small noise on the keyframe values. The first and last stands are never touched.

   Working on keyframes rather than joint angles means the IK works out the joints, so there's no
   abduction-sign bookkeeping. And every augmented motion goes back through the gate like any other.

## Bookkeeping: `datasets/<name>/` on Drive, in new `kflib/dataset.py`

```
datasets/v1/
  motions/<id>.npz      the tracker's format (save_npz), one motion
  motions/<id>.json     source, request, keyframes, attempts, gate report, evaluate(), parent id
  manifest.csv          one row per motion: id, source, kind, duration, ranges, verdict, split
  requests.jsonl        every generated request, kept or not
  split.json            the held-out lists, written once, before any training
```

- **Resumable:** ids already in the manifest are skipped, as in the benchmark's run cell, so a Colab
  disconnect costs nothing.
- **Never overwritten:** a new dataset is a new folder.
- `build_library(ids)` joins motions with `md.concat_clips`, after first **checking** that the root
  position and heading line up at every seam (the Phase 0 caveat). It then calls `save_npz` once and
  writes the library the tracker trains on, with its clip table for scoring per motion.
- `report()` draws the histograms the "done" check asks for: duration, pitch / roll / yaw / height
  ranges, foot lifts, verdict, source.

## Decisions for you

1. **(decide) The mix.**
   - *Recommendation:* ~250 Gemini, ~240 procedural, and the course tricks you don't hold out. Then
     mirror everything (×2) and time-scale a random third.
   - That's ~1,000–1,300 clips, about an hour of motion (~250k frames, ~0.3 GB on the GPU: no problem,
     per Phase 0 question 5).
   - The plan's 10% for course tricks isn't reachable: there are only 7, so they're ~4% by time.
2. **(decide) What's kept.**
   - *Recommendation:* exactly what the gate passes today. That means no IK target off by more than
     2 mm, the kinematic, balance and slide checks pass, and plain replay is HOLDS UP or WOBBLES.
     FALLS is rejected, at least for v1.
3. **(decide) The held-out split, written before any training.**
   - **4a:** the 12 benchmark requests, excluded from generation.
   - **4b:** requests written after training starts (held out automatically).
   - **4c:** course tricks. *Recommendation:* hold out `sneeze`, `push_up` and
     `stand_sit_shake_sit_stand` (short, energetic and long; all have specialists), and train on the
     other four.
   - **Validation:** 10% of generated motions, picked by *request*. A motion's mirror, time-scaled
     and jittered copies always land on the same side as it does; otherwise its mirror would leak the
     test motion into training.
4. **(decide) LLM backend.**
   - *Recommendation:* the free Colab AI. Start with a 20-request pilot to measure its rate limits
     and attempt counts, and switch to AI Studio only if it's too slow.

## Order of work

1. `dataset.py`: write, resume, manifest, split, `build_library` with the seam check, report. Plus
   tests. Laptop, no LLM.
2. `procedural.py` and `augment.py`, plus tests. These run fully on the laptop too, so I can generate
   and gate the procedural half here and give you an acceptance rate and histograms before anything
   touches Colab.
3. The request generator and the Gemini source: a thin `phase2_generate.ipynb` for Colab. Colab AI
   only works inside the notebook, so this part has to run there. It starts with the pilot.
4. Course tricks: import, blend, gate.
5. Split, report, and a 10-motion library npz, ready for Phase 3's smoke test.

**Done when:** ~500 gated motions (before augmentation), a manifest, a fixed split, and histograms that
look broad rather than dominated by one kind of motion.

**One blocker first:** Drive has to mount reliably in Colab, since the dataset lives there. That's the
"folder not found" problem from Phase 1, still unsolved.
