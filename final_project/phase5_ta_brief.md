# For the TA: a general motion tracker for Pupper, and a first robot run

**What it is.** One tracking policy that plays keyframe motions it has never been trained on. Gemini
writes a motion as keyframes, our verifier checks them, and the general tracker plays the result. There is
no per-motion training.

**How it was trained.**
- The course's `Mjlab-Tracking-Flat-Pupper-v3` task, with pupper-mjlab pinned at `aa475e9`.
- Trained on ~978 verified motions: Gemini keyframes, procedural keyframes and the course `sit`, plus
  their mirrors and time-scaled copies, about 78 min of motion. They're joined into one library; the
  task trains it as one long motion.
- **Unchanged from the course config:** the network (256-128-128), the observation (69, no
  history), the action scale, kp 5.5 / kd 0.25, the domain randomization, 4096 envs and the PPO
  settings.
- **One change:** v2 adds one reward term, `paw_height` (`final_project/kflib/rewards.py`). It
  compares each paw's height with the reference's: `exp(-mean squared error / (2 cm)²)`, weight 1.0.
  Without it, the tracker skipped small paw lifts: of 14 test motions with a paw lift, it did a median
  14% of the lift; v2 does 109%.
- v2 is v1 (5,000 iterations) fine-tuned for 5,000 more.
- No limits were changed: the joint limits, gains and speed caps are the course's, and the tilt
  e-stop stays at 60°.

**Simulation results** (`play_motion` scoring; HOLDS UP = ≤ 3 cm and ≤ 10° worst error), on motions
it never trained on:

| | plain replay | v2, zero-shot | specialist |
|---|---|---|---|
| benchmark requests (24) | 10/24 hold up | 22/24 hold up, median 1.1 cm / 3.0° | — |
| new requests (30) | 22/30 | 28/30, median 0.9 cm / 2.5° | — |
| held-out course tricks (3) | 1/3 | 2/3, median 2.5 cm / 6.9° | 2/3, 1.4 cm / 3.0° |

No motion falls in any set. On training motions: 30/30 hold up, median 0.9 cm / 2.5°.

**The file for the robot.** A normal trick folder, `tricks/<name>/` with `policy.json`, `trick.yaml` and
`motion.npz`:
- `policy.json` is made by pupper-mjlab's own `export_trick`: the general weights with one
  held-out motion embedded.
- It's checked to have the same deploy contract as the course's `sneeze` policy: keys, network
  shapes, gains and observation layout.
- `robot/tricks.py` loads it.
- The exact file is scored in simulation before anything else: `phase5_deploy.ipynb`,
  `sim_summary.json` in the folder.

## What we're asking: one thing

**The OK for the first robot runs of this new trained policy** (v2, with the added reward term), as the
course rules require.

- **Which motions:** never-trained motions that first hold up in simulation, as the exact file the
  robot will load.
- **The setup:** the robot on the ground, clear space around it, the student at the robot with a
  hand near the e-stop, and you there.
- **The order:** a gentle motion first.

**How the motions reach the robot.** This uses the robot's existing code only; nothing in `robot/`
changes:
- **Each motion's `policy.json`** is the same general weights plus that motion's reference. We
  checked that the motion block built this way is identical to what `export_trick` writes.
- **Each file goes through the robot's own validator** (`~/pupper_gait_deploy/download_policy.py
  --validate-only`, as `get_trick.py` does) before it's loaded.
- **The controller is loaded through the same `spawn()` path** that `robot_api.switch_to` uses for a
  trick controller the launch missed.
- **Gemini plays it through the normal policy-trick job** (`play_policy_trick`), with the course's
  tilt e-stop (60°).
