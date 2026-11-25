# Live demo: a spoken request → a brand-new motion on Pupper, zero-shot

You ask Gemini Pupper for something it has never done. Gemini calls `prepare_new_motion`. The robot
sends the request to the **motion server** on your laptop, which:
1. writes the keyframes with Gemini and gates the motion;
2. builds v2's `policy.json` with the motion inside;
3. **plays that exact file in simulation.**

Only a motion the policy HOLDS UP on comes back. The robot checks it with its own validator and loads it
as an inactive trick controller. Gemini says it's ready and asks you, and only on your yes does
`play_new_motion` play it. Expect about a minute from the request to "ready".

**Before the first run on the robot: the TA's OK** (`phase5_ta_brief.md`), the robot on the ground,
clear space, you at the robot with a hand near the e-stop.

## Once: the laptop

1. **A Python environment with pupper-mjlab**, pinned. This is a durable copy of what the desk tests used:

   ```powershell
   python -m venv $HOME\pgt-venv
   ```

   ```powershell
   & $HOME\pgt-venv\Scripts\python -m pip install scipy "mjlab @ git+https://github.com/cs123-stanford/pupper-mjlab@aa475e9a393a209b29fd5de7e0ee10f3de643230"
   ```

   If that `pip` form fails, clone pupper-mjlab, run `git checkout aa475e9`, then `pip install -e .`
   inside it.

2. **A template:** any `policy.json` that v2 exported. Run `phase5_deploy.ipynb` once (any held-out
   motion is fine) and download `MyDrive/pupper_general_tracker/robot_tricks/<name>/policy.json`.
   Save it as `final_project\pgt_v2_template.policy.json`. Git ignores `final_project/*.policy.json`, so
   the 1.5 MB of weights stay out of commits. Only its weights are used: every new motion replaces
   the motion inside.

3. **Self-check** (~1 min):

   ```powershell
   cd "$HOME\OneDrive - Stanford\Documents\CS123\Lab 5\gemini_pupper\final_project"; & $HOME\pgt-venv\Scripts\python -m kflib.selftest
   ```

## Each demo: the laptop

1. The Gemini key goes in the environment, **never in a file**:

   ```powershell
   $env:GEMINI_API_KEY = Read-Host "Gemini key"
   ```

2. Start the server. Use a current Gemini Flash model name from AI Studio. Windows asks whether
   Python may accept connections: allow it on **private** networks only.

   ```powershell
   & $HOME\pgt-venv\Scripts\python motion_server.py --template pgt_v2_template.policy.json --model <gemini-flash-model>
   ```

3. Find the laptop's address on the robot's Wi-Fi (the IPv4 address of the Wi-Fi adapter):

   ```powershell
   ipconfig
   ```

## Once: the robot (moves nothing)

1. Get this branch on the robot: the `final-project-kflib` branch has `lab/new_motion.py`, plus one
   import line at the end of `lab/part4_tricks.py`.
2. Tell it where the server is:

   ```bash
   mkdir -p ~/.config/gemini-pupper && echo "http://<laptop-ip>:8765" > ~/.config/gemini-pupper/motion_server
   ```

3. Check the robot can reach the laptop. This queries the laptop's server, not the robot's:

   ```bash
   curl http://<laptop-ip>:8765/health
   ```

4. Restart Gemini Pupper (`robot/stop.sh`, then `robot/start.sh`), at the robot, with your hand near
   the e-stop as always.

## The demo

- **Ask for something no trick does:** "Pupper, lift your back right paw and wave it slowly three
  times."
- **Gemini says it's working it out** (about a minute). The laptop's terminal shows each request:
  `OK` or `NO` plus the reason.
- **Gemini says it's ready and asks.** Say yes, and Pupper plays it, then goes back to standing.
- **If the server refuses** (it didn't hold up in simulation, it tilts too much, or the gate failed),
  Gemini says why. Nothing moves.

Every request is kept on the laptop in `~\pupper_new_motions\new_<n>_<words>\` (keyframes, motion,
policy, simulation result, `result.json`), plus `log.jsonl` for the demo write-up.

## Tests without the robot

```powershell
& $HOME\pgt-venv\Scripts\python test_new_motion.py
```

That's the two robot-side tools on the lab's test rig: 12 checks. `python3 lab/test.py` should still
pass everything it ran before.
