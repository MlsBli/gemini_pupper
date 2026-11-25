"""Final project: brand-new motions, played zero-shot by the general tracker.

Two tools, so a person says yes before Pupper does something it has never done:

    prepare_new_motion(description)  asks the motion server (final_project/motion_server.py,
                                                                      on the laptop) for a motion. The server writes the
                                                                      keyframes with Gemini, checks them, builds the
                                                                      tracker's policy.json with the motion inside, and
                                                                      plays that exact file in simulation: only a motion
                                                                      the policy HOLDS UP on comes back. Here it's checked
                                                                      again (the robot's own validator), loaded as an
                                                                      inactive trick controller, and kept, ready. Nothing
                                                                      moves. Takes about a minute.
    play_new_motion(name)            plays a prepared motion, like any policy trick
                                                                      (play_policy_trick: the trick's controller takes the
                                                                      legs, plays, and hands back to locomotion).

Loading at runtime uses the robot's own path for a controller the launch missed
(spawn_controllers.spawn, as robot_api.switch_to does), with the parameters the
launch gives every trick (tricks.controller_params). Nothing in robot/ changes.

Where the server is: one line, e.g. http://192.168.1.20:8765, in
~/.config/gemini-pupper/motion_server (or the PUPPER_MOTION_SERVER environment variable).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from part3_follow import prepare_to_move
from part4_tricks import play_policy_trick
from robot_api import LOCOMOTION, Result, Robot, with_note
from spawn_controllers import LOADED, controller_states, spawn
from tool_registry import Param, tool
from tricks import Trick, controller_params, load_tricks

SERVER_FILE = Path.home() / ".config" / "gemini-pupper" / "motion_server"
NEW_DIR = Path("/tmp/gemini-pupper/new_motions")  # one folder per new motion (gone after a reboot)
VALIDATOR = Path.home() / "pupper_gait_deploy" / "download_policy.py"  # the controller's own checks (get_trick.py)
PREPARE_TIMEOUT_S = 300
NAME = re.compile(r"^new_\d{3}_[a-z0-9_]{1,24}$")
READY: dict[str, Trick] = {}  # prepared motions, by name, newest last


# ============================================================ what the robot does
def server_url() -> str:
    url = os.environ.get("PUPPER_MOTION_SERVER", "").strip()
    if not url and SERVER_FILE.exists():
        url = SERVER_FILE.read_text().strip()
    return url.rstrip("/")


def ask_server(url: str, description: str) -> dict:
    """POST the request to the motion server; its answer (or why it couldn't be reached)."""
    req = urllib.request.Request(f"{url}/motion", data=json.dumps({"request": description}).encode(),
                                                              headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=PREPARE_TIMEOUT_S) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        return dict(ok=False, reasons=[f"the motion server at {url} didn't answer ({type(e).__name__}): is it running?"])


def course_controller_params() -> dict:
    """The course default controller's settings, which every trick controller starts from
    (as the launch file reads them)."""
    from ament_index_python.packages import get_package_share_directory

    config = Path(get_package_share_directory("neural_controller")) / "launch" / "config.yaml"
    return yaml.safe_load(config.read_text())["neural_controller_default"]["ros__parameters"]


class _NoAliasDumper(yaml.SafeDumper):  # ROS parameter files reject YAML aliases (as in the launch file)
    def ignore_aliases(self, data):
        return True


def install(reply: dict, description: str) -> Trick:
    """Write the server's motion as a trick folder, check it with the robot's validator,
    load it, and spawn its controller inactive. Raises with the reason if a step fails."""
    name = reply["name"]
    if not NAME.match(name):
        raise ValueError(f"unexpected motion name {name!r}")
    d = NEW_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "policy.json").write_text(json.dumps(reply["policy"]))
    (d / "trick.yaml").write_text(yaml.safe_dump(dict(
        description=f"A new motion, never trained on: {description}", type="policy", policy="policy.json",
        start_pose="standing", end_pose="standing", max_body_angle_deg=60)))
    if VALIDATOR.exists():
        check = subprocess.run([sys.executable, str(VALIDATOR), "--validate-only", str(d / "policy.json")],
                                                      capture_output=True, text=True)
        if check.returncode != 0:
            raise ValueError("the robot's policy validator rejected it: " + (check.stdout + check.stderr).strip()[-200:])
    trick = load_tricks(NEW_DIR)[name]
    params = {"type": "neural_controller/NeuralController", **controller_params(trick, course_controller_params())}
    param_file = NEW_DIR / f"{name}.yaml"
    param_file.write_text(yaml.dump({trick.controller: {"ros__parameters": params}}, Dumper=_NoAliasDumper))
    if not spawn([trick.controller], param_file) or controller_states().get(trick.controller) not in LOADED:
        raise RuntimeError(f"couldn't load the controller {trick.controller}; see /tmp/gemini-pupper/robot.log")
    return trick


async def prepare_new_motion(robot: Robot, description: str) -> Result:
    """Make a brand-new motion on the motion server and load it, ready to play. Nothing moves."""
    description = " ".join(description.split())
    if not description:
        return Result(False, "describe the motion: what should Pupper do?")
    url = server_url()
    if not url:
        return Result(False, f"no motion server set: put its address (e.g. http://<laptop-ip>:8765) in {SERVER_FILE}")
    reply = await asyncio.to_thread(ask_server, url, description)
    if not reply.get("ok"):
        return Result(False, "couldn't make that motion safely: " + "; ".join(reply.get("reasons") or ["no reason given"]))
    try:
        trick = await asyncio.to_thread(install, reply, description)
    except Exception as e:  # noqa: BLE001  (Gemini gets the reason)
        return Result(False, f"made the motion, but couldn't load it on the robot: {e}")
    robot.tricks[trick.name] = trick
    robot.leg_controllers.add(trick.controller)  # so every other motion switches it off again
    READY.pop(trick.name, None)
    READY[trick.name] = trick
    sim = reply.get("sim") or {}
    return Result(True, f"new motion '{trick.name}' is ready ({trick.duration:.1f} s long); in simulation the tracker "
                                            f"held it within {sim.get('pos_cm')} cm and {sim.get('tilt_deg')} deg. Nothing has moved yet: "
                                            "ask the user whether to play it now, and only call play_new_motion if they say yes.")


async def play_new_motion(robot: Robot, name: str = "") -> Result:
    """Play a prepared new motion (by default the newest), as a policy trick."""
    if not READY:
        return Result(False, "no new motion is ready: call prepare_new_motion first")
    trick = READY.get(name) if name else READY[list(READY)[-1]]
    if trick is None:
        return Result(False, f"no prepared motion '{name}'; ready: {list(READY)}")
    note = await prepare_to_move(robot)
    job = lambda: play_policy_trick(robot, trick)  # noqa: E731
    return with_note(note, await robot.queue(f"new motion {trick.name}", job, trick.total_time, then=LOCOMOTION))


# ============================================================ what Gemini reads
@tool(
    "prepare_new_motion",
    "Work out a brand-new motion that Pupper has never done, from a description, and get it ready to play. "
    "Use it ONLY when the user asks for a motion that none of your other tools or tricks does. It takes about "
    "a minute, so first tell the user you're working it out and that it takes a minute. It does not move "
    "Pupper: when it's ready, tell the user and ask whether to play it.",
    description=Param("string", "What Pupper should do, in the user's words plus any numbers they gave, "
                                                            "e.g. 'lift the front left paw 4 cm and wave it twice'"),
)
async def prepare_new_motion_tool(robot: Robot, description: str) -> Result:
    return await prepare_new_motion(robot, description)


@tool(
    "play_new_motion",
    "Play a new motion that prepare_new_motion got ready. Call it only after the user says yes to playing it.",
    name=Param("string", "The motion's name from prepare_new_motion; leave empty for the newest one", required=False),
)
async def play_new_motion_tool(robot: Robot, name: str = "") -> Result:
    return await play_new_motion(robot, name)
