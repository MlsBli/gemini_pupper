# Plan: a general motion tracker for Pupper (zero-shot motions)

**Goal.** Someone asks Gemini Pupper for a motion it has never done. Gemini writes it as keyframes, a
verifier checks them, and **one general tracking policy** plays it on the robot, with no training for
that motion. Today every trick needs its own trained policy; this plan replaces that with one policy
trained on thousands of motions.

**Who this is for.** A fresh coding session, or the student, starting from this file alone. It assumes
nothing from earlier conversations. Read "What we already know" first, then work phase by phase. Each
phase ends with a check that says when it is done. Don't skip the checks.

**Ground rules** (from `AGENTS.md`, the course's notes for coding agents; they apply here too):
- The student is learning and has to be able to explain the work. Keep changes reviewable, explain
  what you change and why, and leave the design decisions marked **(decide)** to the student.
- **Never move the robot without the student's explicit OK, right then.** Simulate first. A new
  trained policy needs **a TA's OK before its first run on the robot.** Don't raise limits (joint
  speeds, gains, joint ranges) to make something work. Stop if the motors get hot.
- Don't patch provided infrastructure (pupper-mjlab, `robot/`, the deploy controller) to make a
  symptom go away. Where this plan needs an infrastructure change (Phase 5), it goes through the TA.
- No secrets in files: the Gemini key lives in Colab Secrets or `~/.config/gemini-pupper/api_key`,
  W&B uses `wandb login`.

---

## What we already know (results of the generation benchmark)

The benchmark notebook `final_project/pupper_generation_benchmark.ipynb` compared three ways of turning
a request into a Pupper motion, on 12 requests (precise, style, sequence) × 2 repeats, with the same
verifier, gate and metrics for all:

| | Gemini keyframes | keyframes + UniMate in-betweening | UniMate text-to-motion |
|---|---|---|---|
| passed the gate | **100%** | 83% | 42% |
| precise requests that hit their numbers | **80%** | 50% | 0% |
| jerk (lower = smoother) | **93** | 3,274 | 415 |
| joint travel / body tilt range | **15.4 rad / 26°** | 12.1 / 21° | 5.3 / 2.7° |
| median latency, request → motion ready | 15.8 s | 19.3 s | 12.7 s |
| … of which the LLM call | 15.6 s | 9.6 s | 6.0 s |

**Decision: Gemini keyframes are the generator.** UniMate's own motions are small for Pupper (median
pitch range 1.6° vs 20°). Its best samples are the ones that fail balance, and the retargeting was
measured *not* to be the problem (the IK fit keeps about 98% of the motion). Keyframes' latency is a
single LLM call (the rest of the pipeline takes 0.25 s), so it is fixable separately (Phase 6). Only
54% of keyframe motions "hold up" in plain open-loop physics replay; the rest wobble. **That gap is
what the tracker is for.**

**Reusable code**, all in the benchmark notebook (CPU-only, on top of pupper-mjlab's `motion_design`):
- *The keyframe format* Gemini writes (relative to the stand: body `x, y, dz, roll, pitch, yaw`, feet
  `[dx, dy, dz]`, `ease` stop/flow, `hold`), with `parse_keyframes()` validation and readable errors.
- *Interpolation + IK:* `build_from_keyframes()` (quintic, stop/flow) → `md.Motion` at 50 Hz.
- *The verifier and gate:* `gate()` checks penetration, joint speed ≤ 15 rad/s, joint limits, foot
  slide ≤ 3 cm, balance (centre of mass off the support polygon ≤ 0.3 s), and a fast CPU physics replay
  (`PhysicsGate`, same verdicts as `play_motion`: HOLDS UP / WOBBLES / FALLS). Failures come back as
  sentences Gemini can act on.
- *Reach limits* probed with the course IK (`probe_limits()`), the robot brief, the keyframe system
  prompt with two worked examples, and the LLM backends (Colab AI, which is free, or AI Studio).
- *Metrics:* `evaluate()`, and the precise-request checks (`SPECS`).
- *Storage pattern:* experiments in `runs/<name>/` on Drive, never overwritten.

---

## The course infrastructure this builds on (verify each item in Phase 0)

- **pupper-mjlab** (`https://github.com/cs123-stanford/pupper-mjlab`): MuJoCo Warp + RSL-RL PPO.
  The task is `Mjlab-Tracking-Flat-Pupper-v3`, a BeyondMimic-style tracker. Course notebooks:
  `notebooks/CS123_Pupper_Tricks.ipynb` (design → replay → upload → train), plus the
  `Sit_Stay_Shake`, `Spider` and `UniMate` notebooks.
- **Motion files.** `md.Motion.save_npz()` writes fps, `joint_pos`, `joint_vel`, `body_pos_w`,
  `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `body_names`, plus an optional clip table
  (`clip_names/starts/ends/loops/nexts`). `md.concat_clips()` joins clips whose boundary poses match
  (gap ≤ 0.05 rad).
- **Training** (from the Tricks notebook): `upload_motion` sends the npz to W&B as an artifact, then
  `launch_training(TASK, TrainConfig(env=env_cfg, agent=agent_cfg, registry_name=MOTION_ARTIFACT))`
  with 4096 envs. One motion takes about 15 minutes for 3,000 iterations on a G4 (RTX PRO 6000).
  `policy.json` is uploaded to the run at every checkpoint. Resuming across sessions: check the
  agent's resume option.
- **Scoring:** `python -m mjlab.tasks.tracking.config.pupper.play_motion motion.npz --checkpoint model.pt`
  gives worst body position / tilt error and a verdict. `--viewer viser --share True` lets you watch.
- **Deployed policy contract** (`tricks/<name>/policy.json`, e.g. `tricks/sneeze/policy.json`):
  - observation per step (69): `motion_command` 24, `motion_anchor_ori_b` 6, `base_ang_vel` 3,
    `joint_pos_rel` 12, `joint_vel_rel` 12, `last_action` 12; no history.
  - MLP 256-128-128-12 with ELU, `action_scale` 0.75, kp 5.5, kd 0.25.
  - **the reference motion is embedded in the JSON** (50 Hz), and the robot's launch file loads one
    inactive controller per policy trick.

  The important point: the policy already only sees a reference frame plus its own state. Nothing in
  the network is motion-specific, so a policy trained on many motions can in principle track new ones
  without any change to the contract. **What's missing on the robot is a way to give it a new motion
  at runtime** (Phase 5). `robot/tricks.py`'s `reference` tricks, joint-offset tables streamed over
  the `~/reference_trick` message through the locomotion policy, are a precedent for runtime reference
  input.

---

## Decisions so far

- **2026-11-17, Phase 0 done:** answers with references in `phase0_answers.md`. Phase 1 is planned in
  `phase1_proposal.md`.
  - Colab is the reference environment.
  - pupper-mjlab is pinned to commit `aa475e9`.
- **2026-11-17, scope:** v1 is **quick tricks only**: every motion starts and ends at the stand
  (x = y = 0, yaw = 0), as the plan below assumes. Three things are deferred until the tricks work:
  - **Gaits.** The student wants walking to be zero-shot too, not a pre-trained walking policy, so
    walking waits rather than reusing the course's walking policies.
  - **Tricks that travel or turn.** These need two changes in our code: line each motion up with
    where the previous one ended before joining (a rigid shift and turn of its root path), and let
    the last keyframe be a stand at a new x, y, yaw.
  - **States**, such as a "spider mode". The idea: a state is a rest pose, and keyframes start and
    end at the current state's rest pose instead of the stand.

  Notes and TA questions for all three: the follow-up discussion recorded in `phase0_answers.md`
  (section "Later").
- **2026-11-17, Phase 1 done:** `kflib/` passes its selftest, including the regression: all 24
  benchmark keyframe results rebuild exactly.
- **2026-11-17, Phase 2 decisions** (details in `phase2_proposal.md`):
  - **Mix:** ~250 Gemini + ~240 procedural + course tricks, then mirror everything and time-scale a
    third.
  - **Keep:** gate pass (plain-replay FALLS rejected).
  - **Hold out for 4c:** `sneeze`, `push_up` and `stand_sit_shake_sit_stand`.
  - **Train on `sit` only.** The other recordings fail the gate.
  - **Generated requests skip the held-out tricks,** since the prompt's sneeze example *is* the
    course sneeze.
  - **Validation:** 10%, split by request, with all of a request's copies on one side.
  - **LLM:** Colab AI, with a pilot first.

- **2026-11-18, Phase 2 done:** dataset `v1` on Drive.
  - 1,093 motions (87 min of motion): 471 originals (300 Gemini, 170 procedural, `sit`) plus their
    mirrors and time-scaled copies. Split 978 train / 115 val.
  - The Gemini motions are smaller than the benchmark's. Decided: go on to Phase 3, and let
    Phase 4a show whether that matters.
- **2026-11-18, Phase 3 started:** `phase3_smoke.ipynb` (10-motion smoke test, the course config
  unchanged) and `kflib/score.py` (scores each motion with `play_motion`, plain replay and policy).

- **2026-11-18, Phase 3 smoke test passed:** one policy (course config unchanged, 3,000 iterations,
  10-motion library) tracks **10/10** training motions under `play_motion --checkpoint`, worst errors
  0.8–1.8 cm and 1.5–5.5°. Two motions that only wobbled in plain replay are now tracked: `course_sit`
  (6.5 cm / 12° → 1.7 cm / 2.5°) and `gem_q0251` (20° → 2.7°). So is `proc_00005`, which falls over in
  plain replay. The 3 unseen validation motions also track (3/3), but they're easy (2 of them hold up
  in plain replay anyway).
  - **Finding:** `proc_00005` passed our CPU gate yet FALLS in `play_motion`'s plain replay. The
    fast gate is more forgiving than `play_motion` (it has no latency model), so some motions in
    `v1` are harder than the gate said. That's fine for training: the tracker exists for them.

- **2026-11-18, v1 run set up:** `phase3_v1.ipynb` plus `kflib/runs.py`.
  - **Library:** every training motion of `v1` (~978), in a fixed shuffled order, so seams join
    different motions rather than a motion and its own mirror. The course config is unchanged:
    4096 envs.
  - **Chunks:** training goes in chunks (default 5,000 iterations, up to a target of 20,000). The
    run resumes from the newest checkpoint across all its session folders (mjlab's own lookup
    would fail on an empty newest folder). Checkpoints are saved every 1,000 iterations.
  - **Scoring:** a fixed evaluation set (every validation original + 30 training originals),
    scored with `play_motion` at each chunk.

- **2026-11-24, v1 at 5,000 iterations** (`model_4999.pt`; W&B run rmvk3wq1):
  - **Training sample:** 30/30 HOLDS UP, median 0.9 cm / 2.4°, worst 1.8 cm / 6.8°. That's as good as
    the 10-motion smoke policy (median ~1 cm / ~3°), so the Phase 3 "done" bar is met.
  - **Validation (never trained on):** 47/51 HOLDS UP (plain replay: 28/51), median 1.1 cm / 3.2°.
    3 WOBBLES (`gem_q0156`, `gem_q0210`, `gem_q0236`) and 1 FALLS (`gem_q0137`, 11.4 cm / 28.6°), all
    Gemini motions.
  - Next: train to 10,000, score again, and keep going only while validation improves.

- **2026-11-24, Phase 3 stopped at 5,000 iterations:** the W&B curves are flat and `gem_q0137` looks
  good in the viewer. On to Phase 4.
- **2026-11-24, Phase 4 set up:** `phase4_zeroshot.ipynb`.
  - **Test set `test_v1`:** the 24 benchmark motions (4a; 10 with precision checks) and the 3
    held-out course tricks with their specialists (4c), built locally from the benchmark's
    `results.pkl`. The notebook adds ~30 new requests (4b).
  - **New scorer `kflib/track.py`:** `play_motion`'s scoring loop, plus deployed `policy.json`
    policies (the specialists exist only as these) and the robot's recorded trajectory, so the
    precision checks run on what the robot *did*.
  - **Finding:** `play_motion` isn't bit-for-bit repeatable. Two runs of the same plain replay gave
    5.6 cm / 10.1° and 5.9 cm / 10.4°, and `track.py` falls inside that spread. Verdicts near the
    thresholds can flip, so read ±1–2 motions as noise.

- **2026-11-24, Phase 4 results (v1 at 5,000 iterations, zero-shot).** The self-checks pass (JSON =
  checkpoint; `track.py` = `play_motion`). No motion falls in any set, under any driver.

  | set | plain replay HOLDS UP | v1 HOLDS UP | v1 median | specialist |
  |---|---|---|---|---|
  | 4a benchmark (24) | 10/24, 2.3 cm / 8.1° | **23/24** | 1.0 cm / 3.6° | |
  | 4b new requests (30) | 22/30, 2.2 cm / 4.5° | **29/30** | 0.9 cm / 3.2° | |
  | 4c held-out tricks (3) | 1/3, 3.3 cm / 15.4° | **2/3** | 2.2 cm / 9.0° | 2/3, 1.4 cm / 3.0° |

  - **Precision on the tracked motions:** 8/10 checks hold (crouch depth, lean and look angles,
    nod count). **Both paw-lift checks fail:** the reference lifts front_l 5.0 cm, the robot 0.6 cm.
  - **The student watched the three non-HOLDS-UP motions:** only `stand_sit_shake_sit_stand` really
    fails (the paw barely leaves the ground during the shake); the other two look fine.
  - **Finding: v1 skips small paw lifts, and the verdicts can't see it.**
    - The body-position reward is one exponential over the *mean* squared error of 9 bodies, with a
      10 cm tolerance. A 5 cm paw left on the ground costs about 3% of that one term.
    - The foot termination only fires at an 8 cm miss.
    - HOLDS UP measures the torso only.
    - `kflib.zeroshot.foot_tracking` measures it from the saved trajectories: the share of each
      reference paw lift the robot actually does.

- **2026-11-24, paw lifts measured on the test set:** of 14 test motions that lift a paw, v1 does a
  median **14%** of the lift and mostly skips 12. Plain replay does 73% and skips 1.
- **2026-11-24, v2 set up (the student's choice: a reward term for the four paws, vertical only):**
  - `kflib/rewards.py` `paw_height`: exp(-mean over the 4 paws of the squared height error / std²),
    computed at the paw centres (the `leg_*_3` frames sit at the knees). Tested against the
    kinematics of the reference and of the robot (0.00 mm).
  - `phase3_v2.ipynb` fine-tunes from v1's weights: same library and evaluation set, defaults
    std 2 cm and weight 1.0, target 10,000 iterations (5,000 more).
  - The Phase 4 notebook now compares trackers side by side (`TRACKERS`).

- **2026-11-24, v2 trained to 10,000 iterations** (fine-tuned from v1 with `paw_height`, std 2 cm,
  weight 1.0).
  - **Training sample:** 30/30 HOLDS UP, median 0.9 cm / 2.5° (v1: 0.9 / 2.4).
  - **Validation:** 46/51 HOLDS UP (v1: 47/51), median 1.0 cm / 2.4° (v1: 1.1 / 3.2); 4 WOBBLES and
    1 FALLS (`gem_q0137` again).
  - **The student watched the wobbles:** the paws lift now, but the lifts add some instability.
  - **Decision:** go for an MVP (on to Phase 5) and come back to polish later. Phase 4 with
    `TRACKERS = "pgt_v1,pgt_v2"` picks the tracker that goes forward.

- **2026-11-24, Phase 4, v1 vs v2 (zero-shot), and v2 goes forward:**

  | | plain | v1 @5,000 | **v2 @10,000** |
  |---|---|---|---|
  | 4a benchmark, HOLDS UP / median | 10/24, 2.3 cm / 8.1° | 23/24, 1.0 cm / 3.6° | 22/24, 1.1 cm / 3.0° |
  | 4b new requests | 22/30, 2.2 cm / 4.5° | 29/30, 0.9 cm / 3.2° | 28/30, 0.9 cm / 2.5° |
  | 4c held-out tricks (specialist 2/3, 1.4 cm / 3.0°) | 1/3, 3.3 cm / 15.4° | 2/3, 2.2 cm / 9.0° | 2/3, 2.5 cm / 6.9° |
  | paw lift done (median share; mostly skipped) | 0.73; 1 of 14 | 0.14; 12 of 14 | **1.09; 2 of 14** |
  | precision checks held | | 8/10 | 9/10 (the 10th misses only on hold time, by 0.02 s) |

  No falls anywhere. v2 lifts its paws and tracks tilt and joints more closely; its HOLDS UP
  counts are within noise of v1's.

- **2026-11-24, Phase 5 prepared (MVP path, no robot-software change):**
  - **The path:** a trick folder (`policy.json` + `trick.yaml`) for the robot's `tricks/`, loaded at
    restart. The `policy.json` is pupper-mjlab's `export_trick` on v2's weights with one held-out
    motion embedded.
  - **`kflib/deploy.py`** checks the course trick contract (identical to the sneeze policy's;
    every course trick matches it), the embedded motion, a reference tilt of at most 45° (the
    e-stop stays at 60°), and acceptance by the robot's own `robot/tricks.py`.
  - **`phase5_deploy.ipynb`** exports, simulates the exact file, and zips it.
  - **`phase5_ta_brief.md`** holds the questions for the TA.
  - Nothing runs on the robot without the TA's OK.

## Phase 0: confirm the infrastructure supports this (half a day)

Read, don't assume. Answer each question with the file and line that shows it:
1. How does the tracking task's motion command sample start times? Uniform over the whole motion, or
   BeyondMimic-style adaptive sampling over time bins? Does it respect a clip table, or treat a
   concatenated library as one long motion?
2. What exactly is `motion_command` (24 = reference joint positions + velocities at which step?), and
   what is the "anchor" body?
3. Which domain randomizations does training apply (the summary mentions mass ×0.9–1.3, centre of mass
   up to 1 cm off, friction, PD gains, encoder bias, action latency)?
4. How does `export_trick.py` embed the motion into `policy.json`? Can the same weights ship with a
   different motion (just swapping the embedded motion)?
5. How large a motion file can the tracking env hold on the GPU (frames × bodies)? Is anything
   per-motion precomputed at startup?

**Done when:** each of the five has a written answer with references, and the student agrees with the
data plan in Phase 1 given those answers.

## Phase 1: a reusable keyframe library (1 day)

Move the CPU parts of the benchmark notebook into `final_project/kflib/` (plain Python, importable in
Colab and on a laptop): the keyframe format, `build_from_keyframes`, `gate` and `PhysicsGate`,
`probe_limits`, the robot brief and keyframe prompt, the LLM backends, `evaluate`. Keep behaviour
identical; add small tests: the sneeze example and the course `sit` pass the gate, two paws up for a
second fails it, a too-fast move fails it. These are the benchmark's self-tests 5a.

**Done when:** the tests pass, and a 10-line script turns one request into a gated `md.Motion` and
saves its npz.

## Phase 2: generate the training motions (2–3 days, mostly unattended)

All offline, so the LLM's 15 s per motion doesn't matter. Every motion starts and ends at the stand.
That makes any two motions concatenable with `concat_clips`, which is likely how they go into training
(Phase 0, question 1).

Sources **(decide** the mix; a starting point is suggested**)**:
1. **Gemini keyframes from generated requests (~50%).** Have an LLM write thousands of diverse requests
   across categories: precise (heights, angles, a specific paw, counts, holds), moods and styles, dog
   actions, sequences, body-pose sweeps, small hops. Run each through the keyframe prompt and verifier
   (up to 3 attempts, as in the benchmark). The free Colab AI backend is fine; run several requests in
   parallel threads if its rate limits allow, and keep a log of the requests.
2. **Procedural random keyframes (~40%).** Random but valid keyframe sequences: poses sampled inside
   the probed reach (`probe_limits`), random timings (0.3–1.0 s per move), random holds and paw lifts
   with the lean-away rule. Free and fast, and they cover the joint space more evenly than language
   does. Same gate.
3. **The course's tricks (~10%):** `tricks/sit.py`, `tricks/sneeze/design.py`, spider, and the motions
   embedded in the `tricks/*/policy.json` files. **Hold some out for evaluation** (Phase 4).
4. **Augmentation:** left–right mirroring (careful with abduction signs; the Sit_Stay_Shake notebook's
   `mirror` is the pattern), time scaling ×0.8–1.25 (re-check joint speed), small pose noise on the
   keyframes before IK.

Acceptance **(decide)**: keep motions that pass the kinematic and balance gate. Plain-replay WOBBLES is
fine, since the tracker will tighten it. Think hard before keeping plain-replay FALLS; it's probably
better to reject them at first.

Bookkeeping: one folder per dataset version, `datasets/<name>/` on Drive, holding each motion's npz,
keyframes JSON, request, gate report and source, plus a manifest. **Fix a held-out split before any
training** (Phase 4 needs it).

**Done when:** about 500 gated motions exist (enough for v1), with a manifest and a fixed held-out
split, and a histogram of duration, pitch/roll/height range and foot lifts looks broad, not dominated
by one kind of motion.

## Phase 3: train the general tracker (2–4 days, iterative)

1. **Smoke test first:** concatenate about 10 motions into one library npz, upload, train with the
   unchanged task and config for about 3,000 iterations. Check that it tracks all 10 with
   `play_motion --checkpoint` on each motion's own npz.
2. **v1:** the ~500-motion library. Same observation contract as today, so it stays deployable. Expect
   far more iterations than a single trick; 20–50k is a guess to measure, not a target. Log to W&B,
   keep checkpoints, and plan for Colab sessions ending (resume).
3. **Watch:** per-motion tracking error across the library, not just the mean reward. A generalist
   that is great on easy motions and falls on hard ones looks fine on average.
4. **Only if v1 tracks training motions well but generalizes poorly:** consider (decide, with the TA)
   reference **lookahead** in the observation (e.g. frames +0.1 s and +0.2 s), as universal trackers
   like ABot-C0 and SONIC do. That changes the policy contract and the deploy controller, so it is a
   Phase 5 conversation.

**Done when:** v1 tracks its training motions about as well as a single-trick policy tracks its own
(compare with the trained sneeze or sit policy on the same metric).

## Phase 4: does it generalize? (the zero-shot test, 1 day)

Held-out motions, never trained on:
- **(a)** the 12 benchmark requests' keyframe motions. They are *not* on Drive: the only copy is
  `pupper_generation_benchmark.zip` (downloaded 2026-11-12), whose `results.pkl` holds all 24
  keyframe results (12 requests × 2 repeats).
- **(b)** about 30 new Gemini requests written after training started
- **(c)** held-out course tricks.

For each, compare:
1. plain replay (no policy)
2. the general tracker, zero-shot
3. where one exists, the per-motion specialist (the trained trick policy).

Metrics: success (no fall), worst body position and tilt error (`play_motion --checkpoint`), joint
tracking error, and the benchmark's precision checks on the tracked motion.

**This is the project's headline result:** zero-shot success rate and how close it gets to the
specialists. Also report how the result scales with dataset size (e.g. 100 / 250 / 500 motions); the
ABot-C0 paper reports that unseen-motion performance improves with more training motions.

**Done when:** a table and a short plot exist, and the student can explain where it fails and why.

## Phase 5: on the robot (with the TA; 2–3 days)

1. **Talk to the TA first:** how should a new motion reach the deployed controller at runtime? Options
   to discuss:
   - (i) write a `policy.json` with the general weights plus the new embedded motion, then load or
     activate it; needs runtime controller loading.
   - (ii) stream the reference to a running general controller over a topic, like `~/reference_trick`.
   - (iii) keep a small set of controllers and swap only the motion table.

   This is an infrastructure change and goes through the TA.
2. **First robot run:** a held-out motion that held up well in sim, with the TA's OK, the robot on the
   ground, clear space around it, a hand near the e-stop, and the student's OK right then.
3. Then the full chain: a spoken request → Gemini → keyframes → verifier → the general tracker.

**Done when:** at least one never-trained motion runs on the real robot from a spoken request, on
video.

## Phase 6: latency (parallel track, any time after Phase 1)

Keyframes' latency is one LLM call (15.6 s median). Measure each option, quality against latency, on
the benchmark's 12 requests:
1. **Gemini Live writes the keyframes itself**, as the motion tool's arguments, with no second call.
   This is the biggest lever; check whether quality holds.
2. A lighter or faster model, or less thinking (AI Studio's thinking level).
3. A shorter keyframe format (defaults, fewer fields).
4. (Reach goal) streaming: start the first keyframes while the rest are still being generated.

**Done when:** a quality-vs-latency table exists and a default is chosen.

---

## Timeline (today: 2026-11-17; demo video 12/2, Demo Day 12/3, CoDA B90)

| Days | Work |
|---|---|
| 1 | Phase 0, start Phase 1 |
| 2–4 | Phase 1 done; Phase 2 generation running in the background; Phase 3 smoke test |
| 5–7 | Phase 3 v1 training; Phase 4 evaluation; Phase 6 measurements alongside |
| 8–10 | Phase 5 with the TA; scale data if needed (v2) |
| 11–14 | Integration, the demo video, buffer |

If the time gets tight, the minimum demo is: a held-out motion tracked zero-shot in sim (Phase 4),
plus one on the real robot with the TA (Phase 5).

## Risks

- **The tracking task may not train well on one long concatenated library** (sampling, episode
  length). Phase 0 answers this; the fallback is a small change to the motion command to sample
  per-clip. That's a pupper-mjlab change, so check with the TA.
- **Language-made data clusters** around a few kinds of motion. That's why procedural motions are
  ~40% of the data.
- **Colab sessions end mid-training:** checkpoint to W&B, and know how to resume (Phase 0).
- **Runtime motion loading on the robot** is the critical path for the real-robot demo. Start the TA
  conversation in week 1.

## Prompt for a new session

> I'm a Stanford CS123 student building a general motion tracker for the Pupper robot dog, so it can do
> new motions zero-shot. Read `final_project/general_tracker_plan.md` in this repo first: it has the
> results so far, the course infrastructure it builds on, the plan by phase, and the ground rules
> (also in `AGENTS.md`). The keyframe generator, verifier and gate to reuse are in
> `final_project/pupper_generation_benchmark.ipynb`. Start with Phase 0: read the pupper-mjlab tracking
> task (`https://github.com/cs123-stanford/pupper-mjlab`, task `Mjlab-Tracking-Flat-Pupper-v3`) and
> answer its five questions with references. Then propose how to do Phase 1. Ask me before any
> decision marked "(decide)", and never run anything that moves the real robot.
