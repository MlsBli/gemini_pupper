"""Play one motion in simulation and record what the robot actually did.

New in Phase 4. play_motion (pupper-mjlab) scores a motion but takes only a .pt
checkpoint, and keeps nothing but the worst errors. This is the same scoring loop,
copied from play_motion.main (same env, same nominal robot, same verdict thresholds;
the selftest of Phase 4 checks the numbers match play_motion's), with two additions:

- three ways to drive the robot:
    plain       open-loop joint replay (play_motion without --checkpoint)
    checkpoint  a trained .pt (play_motion --checkpoint)
    json        a deployed trick policy.json (the course's specialist tricks have
                only these), run with pupper-mjlab's json_forward, the NumPy copy of
                the robot's network that the exporter checks parity against
- the robot's trajectory (root and joints) is recorded, so the tracked motion can be
  measured like the reference (e.g. the benchmark's precision checks) and the joint
  tracking error reported.

    python -m kflib.track motion.npz --mode json --policy tricks/sneeze/policy.json --out r.json
Nothing here moves the robot.
"""
import argparse
import concurrent.futures as cf
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# play_motion's constants (pupper-mjlab tasks/tracking/config/pupper/play_motion.py:44-48)
REPLAY_KP, REPLAY_KD = 7.5, 0.25
WOBBLE_POS_M, WOBBLE_TILT_DEG = 0.03, 10.0
FALL_POS_M, FALL_TILT_DEG = 0.10, 45.0


def track(motion_file, mode: str = "plain", policy_path=None) -> dict:
  """Play one motion npz; returns play_motion's numbers plus the robot's trajectory."""
  import torch

  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_ACTION_SCALE
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.tracking.config.pupper.env_cfgs import pupper_tracking_env_cfg
  from mjlab.tasks.tracking.mdp import MotionCommand

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  # --- the env, exactly as play_motion builds it (mass x1, centre of mass not moved)
  cfg = pupper_tracking_env_cfg(motion_file=str(Path(motion_file).resolve()), play=True)
  cfg.scene.num_envs = 1
  cfg.terminations = {}
  for name in ("encoder_bias", "foot_friction", "pd_gains"):
    cfg.events.pop(name, None)
  cfg.events["base_com"].params["ranges"] = {i: (0.0, 0.0) for i in range(3)}
  cfg.events["inertial"].params["alpha_range"] = (0.0, 0.0)
  actuator = cfg.scene.entities["robot"].articulation.actuators[0]
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  if mode == "plain":  # a trained policy keeps the gains it was trained with
    actuator.stiffness, actuator.damping = REPLAY_KP, REPLAY_KD
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("motion")
  assert isinstance(cmd, MotionCommand)
  default = robot.data.default_joint_pos
  n_frames = int(cmd.motion.time_step_total)
  dt = env.step_dt
  wrapped = RslRlVecEnvWrapper(env)

  if mode == "plain":
    def policy(obs):
      del obs
      nxt = torch.clamp(cmd.time_steps + 1, max=n_frames - 1)
      return (cmd.motion.joint_pos[nxt] - default) / PUPPER_ACTION_SCALE
  elif mode == "checkpoint":
    from dataclasses import asdict

    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    runner = MotionTrackingOnPolicyRunner(wrapped, asdict(load_rl_cfg("Mjlab-Tracking-Flat-Pupper-v3")), device=device)
    runner.load(str(policy_path), map_location=device)
    policy = runner.get_inference_policy(device=device)
  elif mode == "json":
    from mjlab.tasks.pupper.export import json_forward

    net = json.loads(Path(policy_path).read_text())
    if net.get("observation_history", 1) != 1:
      raise ValueError(f"{policy_path}: observation_history {net['observation_history']}; trick policies have 1")

    def policy(obs):
      x = obs["actor"].detach().cpu().numpy()
      return torch.as_tensor(json_forward(net, x), dtype=torch.float32, device=device)
  else:
    raise ValueError(f"unknown mode {mode!r}")

  # --- the loop, as in play_motion, recording the robot as it goes
  origin = env.scene.env_origins[0]
  wrapped.reset()
  obs = wrapped.get_observations()
  rec_pos, rec_quat, rec_q, joint_err = [], [], [], []

  def record():
    rec_pos.append((cmd.robot_anchor_pos_w[0] - origin).cpu().numpy())
    rec_quat.append(cmd.robot_anchor_quat_w[0].cpu().numpy())
    rec_q.append(robot.data.joint_pos[0].cpu().numpy())

  record()
  worst_pos, worst_tilt, fell_at, wobble_at = 0.0, 0.0, None, None
  for step in range(n_frames - 1):
    with torch.inference_mode():
      obs, *_ = wrapped.step(policy(obs))
    pos_err = float(torch.linalg.norm(cmd.robot_anchor_pos_w - cmd.anchor_pos_w))
    dot = float(torch.abs((cmd.robot_anchor_quat_w * cmd.anchor_quat_w).sum()))
    tilt = math.degrees(2 * math.acos(min(1.0, dot)))
    worst_pos, worst_tilt = max(worst_pos, pos_err), max(worst_tilt, tilt)
    if fell_at is None and (pos_err > FALL_POS_M or tilt > FALL_TILT_DEG):
      fell_at = (step + 1) * dt
    if wobble_at is None and (pos_err > WOBBLE_POS_M or tilt > WOBBLE_TILT_DEG):
      wobble_at = (step + 1) * dt
    joint_err.append(float(torch.sqrt(torch.mean((robot.data.joint_pos[0] - cmd.joint_pos[0]) ** 2))))
    record()
  env.close()
  verdict = "FALLS" if fell_at is not None else "WOBBLES" if wobble_at is not None else "HOLDS UP"
  return dict(
    verdict=verdict, pos_cm=round(100 * worst_pos, 1), tilt_deg=round(worst_tilt, 1),
    at_s=fell_at if fell_at is not None else wobble_at, joint_rms_rad=round(float(np.mean(joint_err)), 4),
    frames=n_frames, fps=round(1.0 / dt),
    tracked=dict(root_pos=np.array(rec_pos), root_quat=np.array(rec_quat), joint_pos=np.array(rec_q)),
  )


def run(motion_file, mode="plain", policy_path=None, out=None, timeout: float = 1800) -> dict:
  """track() in its own process (a fresh env each time); the result also goes to `out`
  (JSON) and the tracked trajectory next to it (<out>.tracked.npz)."""
  out = Path(out)
  cmd = [sys.executable, "-m", "kflib.track", str(motion_file), "--mode", mode, "--out", str(out)]
  if policy_path:
    cmd += ["--policy", str(policy_path)]
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=Path(__file__).resolve().parents[1])
  except subprocess.TimeoutExpired:
    return dict(verdict=None, error=f"timed out after {timeout:.0f} s")
  if p.returncode != 0 or not out.exists():
    tail = (p.stdout + p.stderr).strip().splitlines()
    return dict(verdict=None, error=tail[-1][:200] if tail else f"exit {p.returncode}")
  return json.loads(out.read_text())


def run_many(jobs: list[dict], out_dir, workers: int = 4) -> list[dict]:
  """jobs: {"id", "npz", "mode", "policy" (optional), ...}; results land in out_dir as
  <id>__<label>.json (already there = skipped, so a stopped run picks up where it left)."""
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  def one(job):
    label = job.get("label", job["mode"])
    out = out_dir / f"{job['id']}__{label}.json"
    r = json.loads(out.read_text()) if out.exists() else run(job["npz"], job["mode"], job.get("policy"), out)
    row = {k: v for k, v in job.items() if k not in ("npz", "policy")}
    row.update(label=label, **{k: v for k, v in r.items()})
    print(f"{job['id']:28} {label:12} {str(r.get('verdict')):9} {r.get('pos_cm')} cm {r.get('tilt_deg')} deg"
          f" joints {r.get('joint_rms_rad')} rad" + (f"  ({r['error']})" if r.get("error") else ""), flush=True)
    return row

  with cf.ThreadPoolExecutor(workers) as ex:
    return list(ex.map(one, jobs))


def load_tracked(result_json) -> "object":
  """The tracked trajectory of a result as an md.Motion (to measure it like the reference)."""
  from .motion import make_motion

  with np.load(str(result_json) + ".tracked.npz") as z:
    return make_motion(z["root_pos"].astype(float), z["root_quat"].astype(float), z["joint_pos"].astype(float))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("motion_file")
  ap.add_argument("--mode", choices=["plain", "checkpoint", "json"], default="plain")
  ap.add_argument("--policy", help="the .pt (checkpoint) or policy.json (json)")
  ap.add_argument("--out", required=True, help="result JSON (the trajectory goes to <out>.tracked.npz)")
  a = ap.parse_args()
  r = track(a.motion_file, a.mode, a.policy)
  tracked = r.pop("tracked")
  np.savez(a.out + ".tracked.npz", **tracked)
  Path(a.out).write_text(json.dumps(r, indent=1))
  print(json.dumps(r))


if __name__ == "__main__":
  main()
