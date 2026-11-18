"""The course's tricks as stand-to-stand motions, for training (and held-out tests).

New in Phase 2. Seven tricks start and end standing: the six recordings below (in
this repo's tricks/<name>/motion.npz) and pupper-mjlab's sit (built from its
keyframes). The recordings aren't perfectly closed: they end 1-4 cm from where
they started, and their stand is 0.5-2 cm taller than ours. So each one is placed
at the origin facing +x (dataset.place), then blended into our stand at both ends
(with_stand_ends), and gated like any other motion. The end may stay a few cm off
the origin; build_library places each motion where the previous one ended, so the
library still has no jump.
"""
from pathlib import Path

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.tricks import sit as course_sit

from .dataset import motion_from_npz, place
from .motion import FPS
from .verify import with_stand_ends

TRICKS_DIR = Path(__file__).resolve().parents[2] / "tricks"  # this repo's tricks/
STAND_TO_STAND = ("pee", "push_up", "sneeze", "stand_downward_dog", "stand_sit_shake_sit_stand",
                  "stand_sit_stand", "sit")
HELD_OUT = ("sneeze", "push_up", "stand_sit_shake_sit_stand")  # Phase 4c (decided 2026-11-17)
TRAIN = tuple(t for t in STAND_TO_STAND if t not in HELD_OUT)


def course_motion(name: str, tricks_dir: Path = TRICKS_DIR) -> md.Motion:
  """A course trick, starting at the origin facing +x, blended into our stand at both ends."""
  if name not in STAND_TO_STAND:
    raise ValueError(f"{name!r} doesn't start and end standing; choose from {STAND_TO_STAND}")
  if name == "sit":
    motion = md.build_motion(course_sit.keyframes(), fps=FPS)
  else:
    motion = motion_from_npz(tricks_dir / name / "motion.npz")
  motion = place(motion, 0.0, 0.0, 0.0)
  motion, _ = with_stand_ends(motion)
  return motion
