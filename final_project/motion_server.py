"""The motion server: a spoken request in, a checked trick policy for the robot out.

Phase 5 (live path). Runs on the laptop, on the same Wi-Fi as Pupper. The robot's
new_motion tool (lab/new_motion.py) sends it a request; it answers with a policy.json
the robot can load, or with why not. For every request:

  1. keyframes   Gemini writes them (kflib's keyframe prompt), the gate checks the
                 motion, up to 3 attempts (kflib.pipeline.request_to_motion)
  2. the file    the general tracker's weights (from TEMPLATE, any exported policy of
                 that tracker) with this motion embedded (kflib.policy_file)
  3. checks      same deploy contract as a course trick, reference tilt at most 45 deg
                 (kflib.deploy)
  4. simulate    the exact file, played by the policy in pupper-mjlab's simulation
                 (kflib.track, json mode): it must HOLD UP, or the robot doesn't get it

    set GEMINI_API_KEY=...    (PowerShell: $env:GEMINI_API_KEY = "...")   never in a file
    python motion_server.py --template pgt_v2_template.policy.json --model <gemini-...-flash>

Nothing here moves the robot. Requests are answered one at a time (the simulator
isn't thread-safe); each takes ~35-60 s (the LLM call ~16 s, the simulation 4-5x the
motion's length on a laptop CPU). Everything is kept in --out (one folder per request).
"""
import argparse
import json
import re
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_REQUEST_CHARS = 300


def slug(text: str, n: int = 24) -> str:
  s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
  return (s[:n].rstrip("_") or "motion")


class MotionMaker:
  """The pipeline, kept warm: imports and the template are loaded once."""

  def __init__(self, template, llm, out_dir, simulate=None):
    from kflib import deploy, policy_file

    self.template = json.loads(Path(template).read_text())
    diffs = deploy.contract_differences(self.template, json.loads(deploy.REFERENCE_POLICY.read_text()))
    if diffs:
      raise ValueError(f"the template differs from a course trick policy: {diffs}")
    self.llm, self.out = llm, Path(out_dir)
    self.out.mkdir(parents=True, exist_ok=True)
    self.simulate = simulate or self._simulate
    self.lock = threading.Lock()
    self.count = len(list(self.out.glob("new_*")))
    self._deploy, self._policy_file = deploy, policy_file

  def _simulate(self, npz, policy_json) -> dict:
    from kflib.track import track

    r = track(npz, "json", policy_json)
    r.pop("tracked", None)
    return r

  def make(self, request: str) -> dict:
    """The whole pipeline for one request; returns what the robot gets back."""
    from kflib.motion import motion_from_dict
    from kflib.pipeline import request_to_motion

    request = " ".join(str(request).split())[:MAX_REQUEST_CHARS]
    if not request:
      return dict(ok=False, reasons=["empty request"])
    with self.lock:
      t0 = time.perf_counter()
      self.count += 1
      name = f"new_{self.count:03d}_{slug(request)}"
      d = self.out / name
      d.mkdir(parents=True, exist_ok=True)
      out = dict(ok=False, name=name, request=request, reasons=[])
      try:
        rec = request_to_motion(request, self.llm, request_id=name)
        out["attempts"] = len(rec.get("attempts") or [])
        if not rec["ok"]:
          out["reasons"] = rec["reasons"]
          return self._done(d, out, t0)
        motion = motion_from_dict(rec["motion"])
        npz = motion.save_npz(d / "motion.npz")
        (d / "keyframes.json").write_text(json.dumps(rec.get("keyframes"), default=float))
        policy = self._policy_file.policy_with_motion(self.template, npz, name)
        (d / "policy.json").write_text(json.dumps(policy))
        tilt = self._deploy.max_tilt_deg(npz)
        if tilt > self._deploy.MAX_REFERENCE_TILT_DEG:
          out["reasons"] = [f"the motion tilts {tilt:.0f} deg; the limit is {self._deploy.MAX_REFERENCE_TILT_DEG:.0f}"]
          return self._done(d, out, t0)
        t1 = time.perf_counter()
        sim = self.simulate(npz, d / "policy.json")
        out.update(sim=sim, sim_s=round(time.perf_counter() - t1, 1), duration_s=policy["trick"]["duration_s"],
                   reference_tilt_deg=round(tilt, 1))
        if sim.get("verdict") != "HOLDS UP":
          out["reasons"] = [f"in simulation the policy {str(sim.get('verdict')).lower()} on it "
                            f"({sim.get('pos_cm')} cm, {sim.get('tilt_deg')} deg): not safe enough to play"]
          return self._done(d, out, t0)
        out.update(ok=True, policy=policy)
        return self._done(d, out, t0)
      except Exception as e:  # noqa: BLE001  (the robot gets a reason, the log the traceback)
        out["reasons"] = [f"{type(e).__name__}: {e}"]
        (d / "error.txt").write_text(traceback.format_exc())
        return self._done(d, out, t0)

  def _done(self, d: Path, out: dict, t0: float) -> dict:
    out["total_s"] = round(time.perf_counter() - t0, 1)
    summary = {k: v for k, v in out.items() if k != "policy"}
    (d / "result.json").write_text(json.dumps(summary, indent=1, default=str))
    with (self.out / "log.jsonl").open("a", encoding="utf-8") as f:
      f.write(json.dumps(dict(summary, time=datetime.now().isoformat(timespec="seconds")), default=str) + "\n")
    print(f"[{summary['total_s']:5.1f} s] {'OK  ' if out['ok'] else 'NO  '} {out.get('name')}: {out['request']!r}"
          + ("" if out["ok"] else f" -> {out['reasons'][:1]}"), flush=True)
    return out


def make_handler(maker: MotionMaker, info: dict):
  class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict):
      data = json.dumps(body, default=str).encode()
      self.send_response(code)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)

    def do_GET(self):
      if self.path == "/health":
        self._send(200, dict(ok=True, **info))
      else:
        self._send(404, dict(ok=False, reasons=["GET /health or POST /motion"]))

    def do_POST(self):
      if self.path != "/motion":
        return self._send(404, dict(ok=False, reasons=["POST /motion"]))
      try:
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(min(n, 10_000)) or b"{}")
      except (ValueError, json.JSONDecodeError):
        return self._send(400, dict(ok=False, reasons=["send JSON: {\"request\": \"...\"}"]))
      self._send(200, maker.make(body.get("request", "")))

    def log_message(self, fmt, *args):  # the per-request line comes from MotionMaker
      pass

  return Handler


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--template", required=True, help="any exported policy.json of the tracker (e.g. from phase5_deploy)")
  ap.add_argument("--model", required=True, help="AI Studio model for the keyframes (key: GEMINI_API_KEY)")
  ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from the robot on this network")
  ap.add_argument("--port", type=int, default=8765)
  ap.add_argument("--out", default=str(Path.home() / "pupper_new_motions"))
  a = ap.parse_args()
  from kflib.llm import LLM

  import importlib

  print("loading (about 15 s) ...", flush=True)
  importlib.import_module("kflib.track")  # pupper-mjlab and the simulator: once, now, not on the first request

  maker = MotionMaker(a.template, LLM("aistudio", a.model), a.out)
  info = dict(template=Path(a.template).name, model=a.model, out=a.out)
  server = ThreadingHTTPServer((a.host, a.port), make_handler(maker, info))
  print(f"motion server on http://{a.host}:{a.port}  (POST /motion, GET /health); results in {a.out}", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()
