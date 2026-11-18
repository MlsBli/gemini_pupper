"""Generated requests: what people might ask Pupper to do, for the Gemini half of the data.

New in Phase 2. An LLM writes short spoken requests by category; each batch is
filtered before anything is generated from it:
  - held out: anything about the Phase 4c test tricks (sneezes, push-ups, sitting
    and shaking a paw). The keyframe prompt's worked example *is* the course sneeze,
    so a "sneeze" request would leak the test motion into training.
  - the benchmark's 12 requests and anything close to them (Phase 4a's tests).
  - near-duplicates of requests already in the pool (same numbers, nearly the same
    words; "lift a paw 3 cm" and "lift a paw 5 cm" are both kept).
Every request, kept or not, goes in the pool file with the reason, so the filter
can be checked by eye.
"""
import difflib
import json
import re
from pathlib import Path

from .metrics import REQUESTS as BENCHMARK_REQUESTS
from .prompt import parse_json

# name -> (share of the requests, what to ask for)
CATEGORIES = {
  "precise": (0.25, "requests with numbers or specifics: a height in cm, an angle in degrees, how many "
                    "times, how long to hold, which paw (front left, back right...), in what order"),
  "mood": (0.20, "moods and styles: act happy, sleepy, nervous, proud, bored, silly, sneaky, shy..."),
  "dog_action": (0.20, "things dogs do in place: play bow, stretch, beg, wag the body, sniff the ground, "
                       "scratch, dig, perk up, shiver, roll the shoulders..."),
  "sequence": (0.15, "two or three actions in a row, said in one request (\"crouch, then look left, then "
                     "wiggle\")"),
  "pose_sweep": (0.15, "body poses and sweeps: lean, tilt, twist, crouch, rise up tall, look around, "
                       "circle the hips, rock forward and back"),
  "hop": (0.05, "small, gentle hops or bounces on the front feet or all feet (tiny, not jumps)"),
}

REQUEST_SYSTEM = """\
You write requests that people might say out loud to Pupper, a small four-legged robot dog
(about 26 cm long, 12 motors, no head, tail or neck: it moves its body and legs). Each request
is something Pupper can do in place in 1.5 to 8 seconds, starting and ending standing: no
walking or running somewhere, no turning around, no lying down or rolling over, no jumping
high, no tricks with objects. Vary the wording, the length and the speaker's tone: some
short ("do a happy wiggle"), some specific ("lift your back right paw 4 cm and hold it two
seconds"), some casual. Every request must be different from the others.

Reply with only the JSON object {"requests": ["...", "..."]}: no prose, no code fences.
"""

# Phase 4c's held-out course tricks (course.HELD_OUT): sneeze, push_up, stand_sit_shake_sit_stand.
HELD_OUT_PATTERNS = {
  "sneeze": r"sneez|achoo|ah+-?choo",
  "push_up": r"push[- ]?ups?\b",
  "stand_sit_shake_sit_stand": r"\bshake\b[^.]*\b(paw|hand|hands)\b|\b(paw|hand)\b[^.]*\bshake\b|shake hands"
                               r"|\bsit\b[^.]*\bshake\b",
}
SIMILAR = 0.8  # difflib ratio at or above this counts as the same request


def _norm(text: str) -> str:
  return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def screen(text: str, pool_texts: list[str]) -> str | None:
  """Why this request can't be used (None = it's fine)."""
  t = _norm(text)
  if len(t) < 6:
    return "too short"
  for trick, pat in HELD_OUT_PATTERNS.items():
    if re.search(pat, text.lower()):
      return f"held out: like the {trick} test trick"
  for b in BENCHMARK_REQUESTS:
    if difflib.SequenceMatcher(None, t, _norm(b["text"])).ratio() >= SIMILAR:
      return f"benchmark: like request '{b['id']}'"
  nums = re.findall(r"\d+(?:\.\d+)?", t)
  for p in pool_texts:  # "paw up 3 cm" and "paw up 5 cm" are different motions: numbers must match too
    q = _norm(p)
    if re.findall(r"\d+(?:\.\d+)?", q) == nums and difflib.SequenceMatcher(None, t, q).ratio() >= SIMILAR:
      return "duplicate"
  return None


def generate_requests(llm, pool_file, n_total: int = 300, batch: int = 25) -> list[dict]:
  """Grow the pool file to n_total usable requests ({"id", "text", "category"}), category
  by category in proportion to CATEGORIES. Resumable: an existing pool is kept and added to.
  Returns the usable requests."""
  pool_file = Path(pool_file)
  pool = [json.loads(line) for line in pool_file.read_text(encoding="utf-8").splitlines()] if pool_file.exists() else []

  def usable():
    return [p for p in pool if not p.get("rejected")]

  for cat, (share, what) in CATEGORIES.items():
    want = round(share * n_total)
    tries = 0
    while sum(p["category"] == cat for p in usable()) < want and tries < 10:
      tries += 1
      have = [p["text"] for p in usable() if p["category"] == cat]
      user = (f"Category: {what}.\nWrite {batch} new requests in this category."
              + (f"\nAlready have (don't repeat these or near-copies): {json.dumps(have[-40:])}" if have else ""))
      text, _, _ = llm(REQUEST_SYSTEM, user)
      try:
        items = [str(x).strip() for x in parse_json(text).get("requests", []) if str(x).strip()]
      except ValueError:
        continue
      with pool_file.open("a", encoding="utf-8") as f:
        for item in items:
          entry = dict(id=f"q{len(pool) + 1:04d}", text=item, category=cat)
          why = screen(item, [p["text"] for p in usable()])
          if why or sum(p["category"] == cat for p in usable()) >= want:
            entry["rejected"] = why or "category full"
          pool.append(entry)
          f.write(json.dumps(entry) + "\n")
  return usable()
