"""One request -> a gated Pupper motion, saved as the tracker's npz. Nothing moves.

    python make_motion.py "lift your front left paw for two seconds" --model <gemini-...-flash>
Then check it in simulation:  python -m mjlab.tasks.tracking.config.pupper.play_motion motion.npz
"""
import argparse

from kflib.llm import LLM
from kflib.motion import motion_from_dict
from kflib.pipeline import request_to_motion

ap = argparse.ArgumentParser()
ap.add_argument("request")
ap.add_argument("--backend", default="aistudio", choices=["aistudio", "colab"])
ap.add_argument("--model", default="", help="required for aistudio")
ap.add_argument("--out", default="motion.npz")
args = ap.parse_args()

rec = request_to_motion(args.request, LLM(args.backend, args.model))
print("PASS" if rec["ok"] else "FAIL", rec["metrics"] and rec["metrics"]["physics"], rec["reasons"])
if rec["ok"]:
    print("wrote", motion_from_dict(rec["motion"]).save_npz(args.out))
