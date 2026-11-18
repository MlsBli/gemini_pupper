"""The verifier and gate: kinematic checks, balance, foot slide and a CPU physics replay.

Moved from the benchmark notebook: cell "Core" (sections "physics gate", "contact /
support analysis", "the gate" and "finishing touches") and finalize from cell "The
three pipelines". One change: the notebook built PHYS = PhysicsGate() when the cell
ran; here physics_gate() builds the same object the first time it is needed, so
importing kflib is quick.
"""
import functools

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import JOINT_NAMES
from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.env_cfgs import get_spec_with_hulls

from .motion import FPS, H, KIN, LEGS, concat_motions, quat_angle_deg, quat_to_rpy, stretch_motion


class PhysicsGate:
  """Open-loop physics replay on the CPU, as play_motion.py does it on the GPU.

  Same model (hull collisions), the MJX-era solver options from
  pupper_constants (4 ms Euler steps, pyramidal cones, 2 iterations), PD
  position servos with play_motion's replay gains (kp 7.5, kd 0.25, 3 Nm),
  targets = the reference's next frame, same verdict thresholds. It skips
  mjlab's sensor/actuator latency model, so it is a little optimistic; section 5
  compares it with play_motion itself. About 20-80 ms per motion.
  """

  KP, KD, EFFORT = 7.5, 0.25, 3.0
  WOBBLE_POS_M, WOBBLE_TILT_DEG = 0.03, 10.0
  FALL_POS_M, FALL_TILT_DEG = 0.10, 45.0

  def __init__(self):
    spec = get_spec_with_hulls()
    o = spec.option
    o.timestep = 0.004
    o.integrator = mujoco.mjtIntegrator.mjINT_EULER
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.impratio = 10.0
    o.iterations = 2
    o.ls_iterations = 5
    o.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    for g in spec.geoms:
      if g.name.endswith("_3_collision"):
        g.priority = 1  # foot friction wins, as in PUPPER_COLLISION
    spec.worldbody.add_geom(
      name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1],
      contype=1, conaffinity=1, condim=3, friction=[0.8, 0.02, 0.01],
    )
    for jn in JOINT_NAMES:
      a = spec.add_actuator(name=f"{jn}_pd", target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
      a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
      a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
      a.gainprm = [self.KP] + [0.0] * 9
      a.biasprm = [0.0, -self.KP, -self.KD] + [0.0] * 7
      a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
      a.forcerange = [-self.EFFORT, self.EFFORT]
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    m = self.model
    self.qadr = np.array(
      [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES]
    )
    self.substeps = int(round((1.0 / FPS) / m.opt.timestep))

  def replay(self, motion: md.Motion) -> dict:
    m, d = self.model, self.data
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = motion.root_pos[0] + np.array([0.0, 0.0, 0.001])
    d.qpos[3:7] = motion.root_quat[0]
    d.qpos[self.qadr] = motion.joint_pos[0]
    d.ctrl[:] = motion.joint_pos[0]
    mujoco.mj_forward(m, d)
    T = len(motion.root_pos)
    traj = np.zeros((T, 19))
    traj[0] = np.concatenate([d.qpos[:7], d.qpos[self.qadr]])
    worst_pos = worst_tilt = 0.0
    fell_at = wobble_at = None
    for i in range(T - 1):
      d.ctrl[:] = motion.joint_pos[i + 1]
      for _ in range(self.substeps):
        mujoco.mj_step(m, d)
      traj[i + 1] = np.concatenate([d.qpos[:7], d.qpos[self.qadr]])
      if not np.isfinite(d.qpos).all():
        fell_at = fell_at or (i + 1) / FPS
        traj[i + 1:] = traj[i]
        break
      pos_err = float(np.linalg.norm(d.qpos[:3] - motion.root_pos[i + 1]))
      tilt = quat_angle_deg(d.qpos[3:7], motion.root_quat[i + 1])
      worst_pos, worst_tilt = max(worst_pos, pos_err), max(worst_tilt, tilt)
      t = (i + 1) / FPS
      if fell_at is None and (pos_err > self.FALL_POS_M or tilt > self.FALL_TILT_DEG):
        fell_at = t
      if wobble_at is None and (pos_err > self.WOBBLE_POS_M or tilt > self.WOBBLE_TILT_DEG):
        wobble_at = t
    verdict = "FALLS" if fell_at is not None else "WOBBLES" if wobble_at is not None else "HOLDS UP"
    return dict(
      verdict=verdict, fell_at=fell_at, wobble_at=wobble_at,
      worst_pos_cm=round(100 * worst_pos, 2), worst_tilt_deg=round(worst_tilt, 1),
      traj=traj.astype(np.float32),
    )


@functools.cache
def physics_gate() -> PhysicsGate:
  """The one PhysicsGate (the notebook's PHYS), built the first time it is needed."""
  return PhysicsGate()


CONTACT_Z = 0.006  # a foot (bottom) or hull point this close to the ground is touching it


def _hull_world_points(d) -> np.ndarray:
  pts = []
  for g, p in KIN.hull_points.items():
    b = KIN.model.geom_bodyid[g]
    pts.append(d.xpos[b] + p @ d.xmat[b].reshape(3, 3).T)
  return np.concatenate(pts)


def _support_margin(com_xy, pts_xy) -> float:
  """Signed distance (m) from the COM to the support polygon; > 0 inside."""
  if len(pts_xy) == 0:
    return float("nan")
  if len(pts_xy) >= 3:
    try:
      hull = ConvexHull(pts_xy)
      return float(-np.max(hull.equations[:, :2] @ com_xy + hull.equations[:, 2]))
    except QhullError:
      pass  # collinear: fall through to segment distance
  # a point or a line: always outside, by the distance to it
  if len(pts_xy) == 1:
    return -float(np.linalg.norm(com_xy - pts_xy[0]))
  a, b = pts_xy[np.argmin(pts_xy[:, 0])], pts_xy[np.argmax(pts_xy[:, 0])]
  ab = b - a
  s = np.clip(((com_xy - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
  return -float(np.linalg.norm(com_xy - (a + s * ab)))


def contact_analysis(motion: md.Motion) -> dict:
  """Per-frame feet, contacts, lowest body point and static support margin."""
  T = len(motion.root_pos)
  feet = np.zeros((T, 4, 3))
  lowest = np.zeros(T)
  margin = np.zeros(T)
  d = KIN.data
  for i in range(T):
    KIN.set_state(motion.root_pos[i], motion.root_quat[i], motion.joint_pos[i])
    feet[i] = [d.site_xpos[KIN.foot_site[leg]] for leg in LEGS]
    hp = _hull_world_points(d)
    lowest[i] = hp[:, 2].min()
    on_feet = feet[i][feet[i][:, 2] - md.FOOT_RADIUS < CONTACT_Z][:, :2]
    on_body = hp[hp[:, 2] < CONTACT_Z][::4, :2]
    margin[i] = _support_margin(d.subtree_com[KIN.base_body][:2], np.concatenate([on_feet, on_body]))
  return dict(feet=feet, lowest_body=lowest, support_margin=margin,
              on=(feet[..., 2] - md.FOOT_RADIUS) < CONTACT_Z)


def slide_stats(feet: np.ndarray, on: np.ndarray, fps: float = FPS) -> dict:
  """Foot sliding: horizontal motion of feet while they touch the ground."""
  v = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1) * fps  # (T-1, 4)
  both = on[1:] & on[:-1]
  worst_span, worst_leg, worst_t = 0.0, None, None
  for k in range(4):
    start = None
    for i in range(len(on) + 1):
      inside = i < len(on) and on[i, k]
      if inside and start is None:
        start = i
      elif not inside and start is not None:
        seg = feet[start:i, k, :2]
        drift = float(np.linalg.norm(seg - seg[0], axis=1).max())
        if drift > worst_span:
          worst_span, worst_leg, worst_t = drift, LEGS[k], (start / fps, (i - 1) / fps)
        start = None
  return dict(
    slide_mean_cm_s=round(100 * float(v[both].mean()) if both.any() else 0.0, 2),
    slide_worst_cm=round(100 * worst_span, 2), slide_worst_leg=worst_leg, slide_worst_t=worst_t,
  )


GATE = dict(
  max_joint_speed=15.0,  # rad/s (AGENTS.md: under about 15)
  max_limit_fraction=0.02,  # joint samples at a limit
  min_lowest=-0.003,  # m: nothing more than 3 mm into the ground
  max_slide_cm=3.0,  # a planted foot may drift at most this much
  max_unbalanced_s=0.3,  # centre of mass outside the support polygon (hops are short)
  allow_wobble=True,  # WOBBLES passes: a trained tracker tightens it up
)


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
  """(length, start) of the longest run of True."""
  best, start, run_start = 0, 0, None
  for i, v in enumerate(list(mask) + [False]):
    if v and run_start is None:
      run_start = i
    elif not v and run_start is not None:
      if i - run_start > best:
        best, start = i - run_start, run_start
      run_start = None
  return best, start


def gate(motion: md.Motion, contacts: dict | None = None) -> dict:
  """Accept/reject a final motion; reasons are written for Gemini to act on."""
  rep = md.check_motion(motion)
  c = contacts or contact_analysis(motion)
  reasons = []
  if rep["lowest_foot_bottom_m"] < GATE["min_lowest"]:
    i = int(np.argmin(c["feet"][..., 2].min(1)))
    reasons.append(f"a foot goes {-100 * rep['lowest_foot_bottom_m']:.1f} cm into the ground around t={i / FPS:.2f} s")
  if rep["lowest_body_point_m"] < GATE["min_lowest"]:
    i = int(np.argmin(c["lowest_body"]))
    reasons.append(f"the body goes {-100 * rep['lowest_body_point_m']:.1f} cm into the ground around t={i / FPS:.2f} s")
  if rep["max_joint_speed_rad_s"] > GATE["max_joint_speed"]:
    jv = np.abs(np.diff(motion.joint_pos, axis=0)) * motion.fps
    i, j = np.unravel_index(np.argmax(jv), jv.shape)
    reasons.append(
      f"joint {JOINT_NAMES[j]} moves at {jv[i, j]:.0f} rad/s around t={i / FPS:.2f} s"
      f" (limit {GATE['max_joint_speed']:.0f}): give that move more time"
    )
  if rep["fraction_of_joint_samples_at_limit"] > GATE["max_limit_fraction"]:
    at = (motion.joint_pos <= KIN.lower + 1e-3) | (motion.joint_pos >= KIN.upper - 1e-3)
    j = int(np.argmax(at.sum(0)))
    i = int(np.argmax(at[:, j]))
    reasons.append(f"joint {JOINT_NAMES[j]} sits at its limit from t={i / FPS:.2f} s: the pose is out of reach")
  s = slide_stats(c["feet"], c["on"])
  if s["slide_worst_cm"] > GATE["max_slide_cm"]:
    a, b = s["slide_worst_t"]
    reasons.append(
      f"foot {s['slide_worst_leg']} slides {s['slide_worst_cm']:.1f} cm on the ground between"
      f" t={a:.2f} and {b:.2f} s: lift it (dz >= 0.02) to move it"
    )
  n_out, i_out = _longest_run(np.nan_to_num(c["support_margin"], nan=-1.0) < -0.01)
  if n_out / FPS > GATE["max_unbalanced_s"]:
    reasons.append(
      f"Pupper is off balance for {n_out / FPS:.2f} s from t={i_out / FPS:.2f} s: its centre of"
      f" mass is up to {-100 * np.nanmin(c['support_margin'][i_out:i_out + n_out]):.1f} cm outside"
      " the feet on the ground. Keep 3 feet down and shift the body ~2 cm away from a foot"
      " before lifting it"
    )
  phys = physics_gate().replay(motion)
  bad = {"FALLS"} if GATE["allow_wobble"] else {"FALLS", "WOBBLES"}
  if phys["verdict"] in bad:
    at = phys["fell_at"] or phys["wobble_at"]
    hint = ""
    out = np.where(c["support_margin"] < -0.005)[0]
    if len(out):
      hint = (
        f"; the centre of mass is outside the feet on the ground from t={out[0] / FPS:.2f} s"
        f" (by up to {-100 * np.nanmin(c['support_margin']):.1f} cm): shift the body away from"
        " lifted feet before lifting them, keep 3 feet down"
      )
    reasons.append(f"in a physics replay the robot {phys['verdict'].lower().replace('falls', 'falls over')} at t={at:.2f} s{hint}")
  return dict(ok=not reasons, reasons=reasons, physics=phys, report=rep, slide=s)


def stand_deviation_cm(motion: md.Motion, i: int) -> float:
  """How far frame i is from a stand (at that frame's x, y, yaw), in cm-ish units."""
  rpy = quat_to_rpy(motion.root_quat[i])
  ref = md.stand(0.0, x=motion.root_pos[i][0], y=motion.root_pos[i][1], yaw=rpy[2])
  got = KIN.feet(motion.root_pos[i], motion.root_quat[i], motion.joint_pos[i])
  feet = max(np.linalg.norm(got[leg] - np.array(ref.feet[leg])) for leg in LEGS)
  return max(100 * feet, 100 * abs(motion.root_pos[i][2] - H), max(abs(rpy[0]), abs(rpy[1])) / 2)


def _pose_keyframe(motion: md.Motion, i: int, t: float) -> md.Keyframe:
  rp, rq, jq = motion.root_pos[i], motion.root_quat[i], motion.joint_pos[i]
  feet = KIN.feet(rp, rq, jq)
  return md.Keyframe(t, tuple(rp), tuple(quat_to_rpy(rq)), feet={k: tuple(v) for k, v in feet.items()})


def with_stand_ends(motion: md.Motion, blend_s: float = 0.6, tol_cm: float = 1.0) -> tuple[md.Motion, int]:
  """Blend from a stand into the motion and back out, where it doesn't already
  start/end standing. Returns the motion and how many frames were prepended."""
  pre = 0
  if stand_deviation_cm(motion, 0) > tol_cm:
    first = _pose_keyframe(motion, 0, blend_s)
    rpy = quat_to_rpy(motion.root_quat[0])
    tr = md.build_motion([md.stand(0.0, x=first.root_pos[0], y=first.root_pos[1], yaw=rpy[2]), first], fps=FPS)
    motion = concat_motions(tr, motion)
    pre = len(tr.root_pos) - 1
  if stand_deviation_cm(motion, -1) > tol_cm:
    last = _pose_keyframe(motion, len(motion.root_pos) - 1, 0.0)
    rpy = quat_to_rpy(motion.root_quat[-1])
    tr = md.build_motion([last, md.stand(blend_s, x=last.root_pos[0], y=last.root_pos[1], yaw=rpy[2])], fps=FPS)
    motion = concat_motions(motion, tr)
  return motion, pre


def auto_slow(motion: md.Motion, max_speed: float | None = None) -> tuple[md.Motion, float]:
  """Stretch time if a joint is too fast (up to 2x)."""
  max_speed = max_speed or GATE["max_joint_speed"]
  speed = md.check_motion(motion)["max_joint_speed_rad_s"]
  if speed <= max_speed:
    return motion, 1.0
  f = min(2.0, speed / (0.9 * max_speed))
  return stretch_motion(motion, f), f


def finalize(motion: md.Motion):
  """Same for every option: blend in/out of a stand if needed, slow down if too fast."""
  motion, pre = with_stand_ends(motion)
  motion, slow = auto_slow(motion)
  return motion, pre, slow
