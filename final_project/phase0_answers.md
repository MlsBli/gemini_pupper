# Phase 0: what the pupper-mjlab tracking task actually does

Answers to the five Phase 0 questions in `general_tracker_plan.md`, read from the code, not assumed.

**Source read:** `cs123-stanford/pupper-mjlab` at commit `aa475e9` (main, 2026-11-05). All paths below
are under `src/mjlab/`, and every reference is a permalink to that commit, so the line numbers stay
valid. The benchmark notebook clones `main` (`BRANCH = "main"`), so if main moves, re-check the lines
that matter (mostly `commands.py`).

Short names used below:

| short | file |
|---|---|
| `commands.py` | `tasks/tracking/mdp/commands.py` (the motion command: loading, sampling, reference state) |
| `env_cfgs.py` | `tasks/tracking/config/pupper/env_cfgs.py` (Pupper's version of the task) |
| `tracking_env_cfg.py` | `tasks/tracking/tracking_env_cfg.py` (the BeyondMimic base config) |
| `export_trick.py` | `tasks/tracking/config/pupper/export_trick.py` (writes `policy.json`) |
| `motion_design.py` | `tasks/tracking/config/pupper/motion_design.py` (`save_npz`, `concat_clips`) |

---

## 1. How are start times sampled? Does it respect a clip table?

**Adaptive (BeyondMimic-style) sampling over 1-second time bins of the whole file. The clip table is
ignored: a concatenated library is trained as one long motion.**

- The default `sampling_mode` is `"adaptive"`
  ([commands.py:598](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L598)).
  Pupper's training config doesn't change it; only `play=True` switches to `"start"`
  ([env_cfgs.py:232](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L232)).
- **Bins:** `bin_count = time_step_total // (1 / step_dt) + 1`
  ([commands.py:93](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L93)).
  The step is 5 × 4 ms = 20 ms, so that's 50 frames per bin: **one bin per second of the whole file**.
- **Sampling** ([commands.py:249-290](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L249-L290)):
  when an episode ends by *termination* (a fall, not a time-out), its current bin gets a failure count.
  Each bin's probability is its smoothed failure count plus a uniform floor of
  `adaptive_uniform_ratio / bin_count` (0.1, [commands.py:596](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L596)).
  A bin is drawn, then a uniform point inside it, so a start can fall anywhere, including mid-motion.
  The failure counts are an exponential moving average with α = 0.001 per step
  ([commands.py:419-424](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L419-L424)).
  The kernel size is 1, so there's no smoothing across bins
  ([commands.py:594](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L594)).
- **When it samples:** at every episode reset (`CommandManager.reset → _resample → _resample_command`,
  [managers/command_manager.py:99-122](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/managers/command_manager.py#L99-L122)),
  and mid-episode only when the reference runs off the end of the file
  ([commands.py:407-415](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L407-L415)).
  The time-based resample is disabled (`resampling_time_range = 1e9`,
  [tracking_env_cfg.py:142](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L142)).
  Episodes last 10 s ([tracking_env_cfg.py:312](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L312)).
  So an episode plays forward from its start point **straight across clip boundaries** into the next
  motions of the library.
- **Clip table ignored in training:** `MotionLoader` reads only the six state arrays
  ([commands.py:40-60](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L40-L60)),
  never `clip_*` (and not `fps` either: it assumes the file is at the 50 Hz control rate). The clip
  table is used only at export, for the robot
  ([export_trick.py:90-96](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/export_trick.py#L90-L96)).

**What this means for us.**
- A library of motions that **all start and end at the stand** trains as-is, with no change to
  pupper-mjlab. Crossing a boundary is just "stand, then the next motion". That's also a useful thing
  to learn, because the robot will do exactly that.
- Time is sampled uniformly (plus extra on the hard seconds), so a 6 s motion gets 3× the samples of a
  2 s one. That's mild and probably fine; Phase 3's per-motion errors will show if not.
- **Caveat found here:** `concat_clips` only checks that the *joints* match at a boundary (gap
  ≤ 0.05 rad, [motion_design.py:517-523](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/motion_design.py#L517-L523)).
  It copies each clip's root position and heading unchanged
  ([motion_design.py:537-543](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/motion_design.py#L537-L543)).
  A clip that ends away from x = y = 0 or yaw = 0 therefore makes the reference **teleport** at the
  next boundary. The global root-position reward is in world coordinates
  ([tracking/mdp/rewards.py:26-33](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/rewards.py#L26-L33)),
  and the position termination checks height only
  ([tracking/mdp/terminations.py:27-34](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/terminations.py#L27-L34)).
  So a jump wouldn't end the episode; it would quietly poison the reward for a while.
  - Gemini keyframes are safe: `parse_keyframes` forces the first and last keyframe to be the stand
    with |x|, |y| ≤ 5 mm and yaw = 0.
  - Procedural motions must follow the same rule.
  - Course tricks that travel or turn (the spider walk) have to be shifted back to the origin before
    joining.
  - Phase 2 should check root position and heading at every boundary, not just the joints. We can do
    that in our own library-building code, with no pupper-mjlab change.

## 2. What exactly is `motion_command`, and what is the anchor?

**`motion_command` (24) = the reference's joint angles (12, absolute, not relative to the default
pose) and joint velocities (12), at the frame the robot should reach at the end of the coming 20 ms
control step. No lookahead beyond that. The anchor is `base_link`, the torso.**

- `command = cat(joint_pos, joint_vel)` at `time_steps`
  ([commands.py:125-135](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L125-L135)),
  fed to the actor through `generated_commands`
  ([tracking_env_cfg.py:48-50](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L48-L50)).
  The joint angles come straight from the npz's `joint_pos`.
- **Which step:** in `env.step`, physics runs first, then terminations and rewards (scored against
  frame *k*), then `command_manager.compute` advances `time_steps` to *k+1*, and only then are
  observations computed
  ([envs/manager_based_rl_env.py:421-464](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/envs/manager_based_rl_env.py#L421-L464);
  the increment is at [commands.py:408](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L408)).
  So the policy sees frame *k+1*, acts for 20 ms, and is then scored against that same frame *k+1*.
  The only hint about the future is the reference joint *velocity*.
- **Anchor:** `anchor_body_name = "base_link"`
  ([env_cfgs.py:165](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L165)).
  `motion_anchor_ori_b` (6) is the reference torso orientation as seen from the robot's own torso
  frame, given as the first two columns of the rotation matrix
  ([observations.py:31-41](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/observations.py#L31-L41)).
  It's a full relative rotation, so it includes yaw. It gets the same 0–1 step IMU lag as the angular
  velocity ([env_cfgs.py:198-200](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L198-L200)).
- The actor gets **no position information**. `motion_anchor_pos_b` and `base_lin_vel` are removed
  ([env_cfgs.py:188-190](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L188-L190)),
  but the critic keeps them, plus the bodies' poses
  ([tracking_env_cfg.py:82-107](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L82-L107),
  [env_cfgs.py:201-203](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L201-L203)).
- Order and sizes match the deploy contract: 24 + 6 + 3 + 12 + 12 + 12 = 69
  ([export_trick.py:40-48](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/export_trick.py#L40-L48)).
  The exporter refuses any other layout
  ([export_trick.py:58-64](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/export_trick.py#L58-L64)).

**What this means for us.** The plan's premise holds: nothing in the observation is specific to one
motion. The policy sees one target frame plus its own state. One frame with no lookahead is the
generalist's main handicap on fast moves and hops, and that is the Phase 3.4 question, unchanged.
Open question for Phase 5 (TA): the observation includes the yaw difference, so how does the deploy
controller line up the reference's heading with the robot's when a trick starts?

## 3. Which domain randomizations does training apply?

The summary was close, with one correction: **action latency is fixed, not randomized.**

| What | Range | When | Reference |
|---|---|---|---|
| Mass and inertia, every link | ×0.9–1.3 | once per env, at startup | [env_cfgs.py:211](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L211) → [pupper_env_cfg.py:366-375](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/pupper/pupper_env_cfg.py#L366-L375) |
| Torso centre of mass | ±1 cm in x, y, z | startup | [env_cfgs.py:209-216](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L209-L216) |
| PD gains | kp ×0.6–1.1, kd ×0.8–1.5 (nominal kp 5.5, kd 0.25) | startup | [env_cfgs.py:210](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L210) → [pupper_env_cfg.py:353-363](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/pupper/pupper_env_cfg.py#L353-L363) |
| Foot friction (all four feet share one value) | 0.3–1.2 | startup | [tracking_env_cfg.py:193-202](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L193-L202), feet chosen at [env_cfgs.py:208](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L208) |
| Encoder bias | ±0.01 rad | startup | [tracking_env_cfg.py:185-192](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L185-L192) (the joint observation reads it: `biased=True`, [line 74](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L74)) |
| Pushes (sets the base velocity) | every 1–3 s; ±0.2 m/s in x and y, ±0.1 in z, ±0.3 rad/s | interval | [tracking_env_cfg.py:166-171](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L166-L171), ranges at [env_cfgs.py:176-184](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L176-L184) |
| Start state | pose ±2 cm in x and y, ±5 mm in z, ±0.1 rad roll and pitch, ±0.2 rad yaw; velocity as for pushes; joints ±0.1 rad | each reset | [env_cfgs.py:168-183](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L168-L183), [tracking_env_cfg.py:153](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L153), applied at [commands.py:333-365](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L333-L365) |
| IMU latency (angular velocity and anchor orientation) | 0 or 1 control step, 50/50 | every step | [env_cfgs.py:191-200](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L191-L200) |
| Actuator delay | **fixed** 16 ms (4 physics steps), *not random* | always | [env_cfgs.py:128-138](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L128-L138), why: [latency.py:38-53](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/pupper/mdp/latency.py#L38-L53) |
| Observation noise (actor only) | anchor orientation ±0.05, angular velocity ±0.2, joint angles ±0.01, joint velocities ±0.5 | every step | [tracking_env_cfg.py:56-79](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/tracking_env_cfg.py#L56-L79), [env_cfgs.py:193-206](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L193-L206) |

"Startup" means sampled **once per env when the env is created**, not every episode
([managers/event_manager.py:109](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/managers/event_manager.py#L109)).
With 4096 envs that still spreads widely.

Not randomized: the ground (flat plane), the torso and leg collision hulls' friction (fixed at 0.8),
motor torque limits, and the actuator delay.

`play=True` turns off pushes, noise and start-state randomization
([env_cfgs.py:226-232](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L226-L232)).
`play_motion` also pins the robot to its nominal values so every verdict is repeatable
([play_motion.py:76-85](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/play_motion.py#L76-L85)).

## 4. How does `export_trick.py` embed the motion? Can the same weights ship with a different motion?

**Yes, with no code change. The weights and the motion come from two independent inputs.**

- The weights come from the checkpoint
  ([export_trick.py:123-127](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/export_trick.py#L123-L127)).
  `export_pupper_policy_from_env` folds the observation normalizer into the first layer and reads only
  robot constants from the env: joint limits, kp/kd, the default pose and the action scale
  ([tasks/pupper/export.py:254-292](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/pupper/export.py#L254-L292)).
  Nothing there depends on the motion.
- The motion comes from whatever `--motion-file` the env was built with
  ([export_trick.py:119](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/export_trick.py#L119)).
  The JSON gets a `trick` block (name, fps, frames, duration, plus the clip table if the npz has one)
  and a `motion` block. The `motion` block holds **only three arrays per 50 Hz frame**: `joint_pos`
  (12), `joint_vel` (12) and the anchor quaternion `anchor_quat_wxyz` (4)
  ([export_trick.py:74-101](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/export_trick.py#L74-L101)).
  The robot needs nothing else from a motion.
- So `export_trick --checkpoint general.pt --motion-file new.npz --name new --out new.json` ships the
  general weights with a new motion. The parity check
  ([export_trick.py:130-147](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/export_trick.py#L130-L147))
  reruns on the new motion. It verifies the network's arithmetic, not tracking quality; that's
  `play_motion --checkpoint`'s job.
- Since the weights block doesn't depend on the motion, a runtime path could also just **replace the
  `motion` block** (and `trick.frames` / `trick.duration_s`) of an existing JSON, without building an
  env. That's relevant to Phase 5 options (i) and (iii). It's cheap to confirm later by diffing two
  exports of the same checkpoint.

**Side effect during training:** the trick runner exports and uploads `policy.json` at every
checkpoint save, every 500 iterations
([runner.py:38-40](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/runner.py#L38-L40),
[rl_cfg.py:39](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/rl_cfg.py#L39)).
When training on a library, that JSON embeds the **whole library**: about 28 numbers ≈ 280 bytes per
frame, so roughly 28 MB for 100k frames. It's wasteful but harmless (a failed export never stops
training, [runner.py:65-66](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/runner.py#L65-L66)).
That run-level `policy.json` isn't meant to be deployed. Deploy JSONs are always re-exported with one
motion.

## 5. How large a motion file fits on the GPU? What is precomputed per motion at startup?

**Memory is not the limit. Only the bins are precomputed, and nothing is per-clip.**

- `MotionLoader` copies all six arrays to the GPU as float32, for all 13 bodies in the npz. It then
  keeps a *second*, indexed copy of the body arrays for the 9 tracked bodies
  ([commands.py:40-59](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L40-L59);
  tracked bodies at [env_cfgs.py:74-78](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/env_cfgs.py#L74-L78);
  13 bodies checked in `tricks/sit.npz`).
- Per frame that is 24 joint values + 13 × 13 body values + 9 × 13 tracked copies = 310 floats ≈
  **1.24 KB/frame**. On disk the npz (`np.savez`, uncompressed) is ≈ 0.77 KB/frame.

| library | frames (at ~4 s per motion) | GPU | npz |
|---|---|---|---|
| 500 motions (v1) | 100k | ~125 MB | ~77 MB |
| 5,000 motions | 1M | ~1.2 GB | ~0.8 GB |

The G4's RTX PRO 6000 has 96 GB, so the motion table never matters. Env count and physics dominate.

- **Precomputed at startup:** only the time bins (`bin_count`, failure counters, kernel;
  [commands.py:93-104](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L93-L104)).
  That's 2,001 bins for 2,000 s. There are no per-clip structures and no normalization statistics.
- **Precomputed offline** in `save_npz`, on the CPU: body poses by forward kinematics, and velocities
  by central finite differences
  ([motion_design.py:305-350](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/motion_design.py#L305-L350)).
  It's a Python loop over frames, so save a library **once, after `concat_clips`**: velocities at the
  boundaries then come out right, and it's one pass instead of one per motion.
- **Constraints the loader assumes but doesn't check:**
  - The npz is at 50 Hz: `fps` is never read.
  - Its body order matches the robot entity's: tracked bodies are indexed by the robot's body indexes
    ([commands.py:75-79](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L75-L79)).

  Both hold as long as every npz is made by `md.Motion.save_npz` from 50 Hz motions, as the keyframe
  pipeline does.

---

## Bonus findings the later phases need

- **Resuming after a Colab disconnect** works out of the box. Set `agent.resume = True` plus
  `wandb_run_path` (and optionally `wandb_checkpoint_name`), and `train` downloads the checkpoint from
  W&B ([scripts/train.py:119-138](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/scripts/train.py#L119-L138),
  [rl/config.py:111-117](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/rl/config.py#L111-L117)).
  Checkpoints are saved every 500 iterations, and `max_iterations` defaults to 5,000
  ([rl_cfg.py:39-41](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/rl_cfg.py#L39-L41)),
  so a long run needs that raised.
- **Zero-shot scoring needs no code change.** `play_motion motion.npz --checkpoint model.pt` builds the
  env from whatever npz you give it
  ([play_motion.py:74](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/config/pupper/play_motion.py#L74)),
  so a held-out motion is just another file.
  - It runs one env per call, so scoring hundreds of motions one by one is slow. Phase 3/4 will want
    our own batched evaluator in `final_project/`: load the library once, and start each env at a
    clip's first frame with `MotionCommand.reset_to_frame`
    ([commands.py:567-582](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L567-L582)).
    That uses pupper-mjlab without changing it.
  - Its position error is in world coordinates, and its "tilt" is the full rotation difference
    (yaw included).
- **Specialists to compare against** already exist in this repo, as `tricks/*/policy.json`: jump,
  lie_downward_dog, lie_sit_lie, pee, push_up, sit, sneeze, spider, stand_downward_dog. Each embeds
  its reference motion, so those motions can also be held-out test motions (Phase 4c).
- **The benchmark's `PhysicsGate` is not the training physics.** It uses MJX-era solver options
  (2 iterations, pyramidal cones), replay gains kp 7.5, and no latency model. It's a fast filter; the
  tracker's scores always come from `play_motion`.

## Does the Phase 2 data plan still hold?

Yes, with three additions. Phase 2 itself (the source mix, the acceptance rule) is still yours to
decide.
1. **Library assembly checks root continuity**, not just joints: every motion starts and ends at the
   stand at x = y = 0 and yaw = 0 (within 5 mm / 1°). Course tricks that travel get shifted back to
   the origin first.
2. **Build the library with `concat_clips`, then call `save_npz` once.** The clip table is ignored by
   training but kept in the file, so we can score per motion later.
3. **Keep motions at 50 Hz.** Time-scaling augmentation must resample back to 50 Hz, as the notebook's
   `stretch_motion` does.

---

## Later: travel, states and gaits (deferred 2026-11-17)

Not in v1. Kept here so nothing has to be worked out again.

- **Tricks that travel or turn.** Training handles a library whose path wanders, with no pupper-mjlab
  change:
  - each reset puts the robot exactly on the reference
    ([commands.py:328-341](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L328-L341));
  - the body rewards are measured relative to the robot's own position and heading
    ([commands.py:396-405](https://github.com/cs123-stanford/pupper-mjlab/blob/aa475e9a393a209b29fd5de7e0ee10f3de643230/src/mjlab/tasks/tracking/mdp/commands.py#L396-L405)).

  Two changes are needed in our code:
  1. When joining, line each motion up with where the previous one ended (shift and turn its root
     path; the joints stay the same).
  2. Let the keyframe format end at a stand with a new x, y, yaw, and measure foot moves from where
     each foot is currently planted.

  On the robot, travel is open loop: the policy has no position sensing.
- **States** (spider mode). The course's interactive tricks already are states:
  - clips, with a looping idle clip so Pupper stays in the stance;
  - `exit_clip` to get back to standing;
  - `start_pose` / `end_pose` in `trick.yaml`;
  - see `robot/tricks.py:26-58` and pupper-mjlab `tricks/spider.py`.

  For the general tracker, a state is a rest pose (for the spider, `SPIDER_POSE` at `SPIDER_HEIGHT`).
  Keyframes start and end at the current state's rest pose. The tool layer remembers which state
  Pupper is in; the tracker itself doesn't need to know.
- **Gaits.** These are deferred until the quick tricks work. The goal is zero-shot walking too, not
  reusing the course's pre-trained walking policies.
- **Questions for the TA:**
  1. Does the trick controller line the reference up with the robot's heading only when a trick
     starts, or at every clip switch too? The controller (`pupper_gait_deploy`, `trick_motion.hpp`)
     isn't in this repo.
  2. Trick controllers first hold the default (standing) pose and then fade in (`robot/tricks.py:73-76`).
     Can a controller start, or take a new motion, while Pupper is already in a non-standing pose?
