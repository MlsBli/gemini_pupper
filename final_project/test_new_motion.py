"""Desk test for the new_motion tools (lab/new_motion.py): no robot, no ROS, no Gemini.

    python final_project/test_new_motion.py     (from the repo root)

Uses the lab's own test rig (robot/test_tools.py): the real Robot and tool registry on
a fake robot and a fake clock. The motion server, the robot's policy validator and the
controller spawner are stood in for (the server returns a real trick policy file, the
course's sneeze, as if it were a new motion). Checks: the two-step flow (prepare moves
nothing; play does, and only after prepare), failures come back as reasons, the new
controller is loaded and then switched off by other motions, and Gemini sees both tools.
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "robot"))
import test_tools as tt  # noqa: E402  (installs the fake ROS)

sys.path.append(str(REPO / "lab"))
import new_motion as nm  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
  print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
  RESULTS.append(bool(ok))


def fake_server(policy: dict, ok: bool = True, reason: str = ""):
  def ask(url, description):
    if not ok:
      return dict(ok=False, name="new_001_x", reasons=[reason])
    return dict(ok=True, name="new_001_wave_a_paw", policy=policy, sim=dict(verdict="HOLDS UP", pos_cm=1.1, tilt_deg=2.9))
  return ask


async def main():
  tmp = Path(tempfile.mkdtemp())
  nm.NEW_DIR = tmp / "new_motions"
  nm.VALIDATOR = tmp / "no_validator_here.py"  # the robot's validator only exists on the robot
  nm.course_controller_params = lambda: {"model_path": "", "init_duration": 0.5}
  nm.server_url = lambda: "http://laptop:8765"
  policy = json.loads((REPO / "tricks" / "sneeze" / "policy.json").read_text())
  rig = tt.Rig()

  def fake_spawn(names, param_file):  # the robot's spawner loads controllers inactive
    for n in names:
      rig.robot.controllers[n] = "inactive"
    fake_spawn.files.append(Path(param_file).read_text())
    return True
  fake_spawn.files = []
  nm.spawn = fake_spawn
  nm.controller_states = lambda: dict(rig.robot.controllers)

  tools = {t["name"] for t in tt.tools.declarations(rig.ts)}
  check("Gemini sees prepare_new_motion and play_new_motion", {"prepare_new_motion", "play_new_motion"} <= tools)

  res = await rig.call("play_new_motion")
  check("play before prepare: refused with a reason", not res.ok and "prepare_new_motion" in res.message, res.message)

  nm.ask_server = fake_server(policy, ok=False, reason="in simulation the policy falls on it")
  res = await rig.call("prepare_new_motion", description="do a backflip")
  check("a motion the server rejects: refused, with the server's reason", not res.ok and "falls" in res.message, res.message)

  nm.server_url = lambda: ""
  res = await rig.call("prepare_new_motion", description="wave a paw")
  check("no server address: says where to put it", not res.ok and "motion_server" in res.message, res.message)
  nm.server_url = lambda: "http://laptop:8765"

  nm.ask_server = fake_server(policy)
  before = rig.robot.active()
  res = await rig.call("prepare_new_motion", description="wave a paw")
  check("prepare: ready, and tells Gemini to ask the user first", res.ok and "ask the user" in res.message, res.message)
  check("prepare moves nothing (the same controllers are on)", rig.robot.active() == before, f"{before} -> {rig.robot.active()}")
  ctrl = "trick_new_001_wave_a_paw"
  check("its controller is loaded, inactive", rig.robot.controllers.get(ctrl) == "inactive", str(rig.robot.controllers.get(ctrl)))
  check("the spawner got the trick's policy and the controller type",
        fake_spawn.files and "new_001_wave_a_paw/policy.json" in fake_spawn.files[-1].replace("\\", "/")
        and "neural_controller/NeuralController" in fake_spawn.files[-1], fake_spawn.files[-1][:200] if fake_spawn.files else "")
  check("it's a trick now, and other motions switch it off", ctrl in rig.ts.leg_controllers and "new_001_wave_a_paw" in rig.ts.tricks)

  start = len(rig.records)
  res = await rig.call("play_new_motion")
  out = " | ".join(rig.outcomes(start))
  check("play: accepted", res.ok, res.message)
  check("play: the new controller took the legs, then locomotion is back", "back in locomotion" in out, out)

  res = await tt.tools.call(rig.ts, "play_new_motion", {"name": "new_999_nope"})  # (rig.call's own 1st arg is "name")
  await rig.settle()
  check("an unknown name: refused, listing the ready ones", not res.ok and "new_001_wave_a_paw" in res.message, res.message)

  print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
  return all(RESULTS)


if __name__ == "__main__":
  sys.exit(0 if asyncio.run(main()) else 1)
