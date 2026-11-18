"""kflib: the keyframe motion generator and verifier from the generation benchmark.

The CPU half of pupper_generation_benchmark.ipynb (option 1, Gemini keyframes) as
plain Python, so Colab notebooks and scripts can import it. Behaviour is the
notebook's: the code was copied, not rewritten (each module's docstring says from
which cell and what changed).

  motion     rotations, md.Motion helpers, feet, dict round trip
  keyframes  the keyframe format Gemini writes: parse, interpolate, IK
  verify     the gate: kinematics, balance, foot slide, CPU physics replay
  metrics    evaluate() and the benchmark's 12 requests with their checks
  prompt     what the LLM is told (measured reach, format, examples)
  llm        the LLM backends ("colab", "aistudio") and a stopwatch
  pipeline   request_to_motion(): request in, gated motion out

Needs pupper-mjlab installed, pinned to the commit Phase 0 read:
  pip install -e "git+https://github.com/cs123-stanford/pupper-mjlab@aa475e9a393a209b29fd5de7e0ee10f3de643230#egg=mjlab"
Check it with:  python -m kflib.selftest   (from final_project/)

Nothing here moves the robot: it writes motions (md.Motion, .npz files).
"""

PUPPER_MJLAB_COMMIT = "aa475e9a393a209b29fd5de7e0ee10f3de643230"
