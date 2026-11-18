"""The LLM backends that write keyframes, and a stopwatch for latency.

Moved from the benchmark notebook: LLM from cell "LLM backend" and Stopwatch from
cell "The three pipelines". Changes:
- "aistudio" gets its key from gemini_key(): Colab Secrets in Colab, otherwise the
  GEMINI_API_KEY environment variable or ~/.config/gemini-pupper/api_key. The key is
  never printed and never written anywhere.
- Stopwatch takes the LLM it times (the notebook used a global `llm`).

backends:
  "colab"     google.colab.ai: free with Colab, no key; only inside a Colab notebook.
  "aistudio"  the Gemini API with your AI Studio key; needs a model name.
"""
import contextlib
import inspect
import os
import time
from pathlib import Path
from time import perf_counter

KEY_FILE = Path.home() / ".config" / "gemini-pupper" / "api_key"


def gemini_key() -> str:
  """The AI Studio key: Colab Secrets (GEMINI_API_KEY), else the environment, else KEY_FILE."""
  try:
    from google.colab import userdata
  except ImportError:
    userdata = None
  if userdata is not None:
    try:
      key = userdata.get("GEMINI_API_KEY")
      if key:
        return key
    except Exception:  # noqa: BLE001  (secret not set, or notebook access off)
      pass
  key = os.environ.get("GEMINI_API_KEY", "").strip()
  if key:
    return key
  if KEY_FILE.exists():
    return KEY_FILE.read_text().strip()
  raise RuntimeError(
    "No Gemini key: add GEMINI_API_KEY to Colab Secrets (with notebook access), "
    f"or set the GEMINI_API_KEY environment variable, or put it in {KEY_FILE}"
  )


class LLM:
  """One text-in, text-out call; returns (text, seconds, seconds_waited_on_rate_limits)."""

  def __init__(self, backend: str, model: str = ""):
    self.backend = backend
    if backend == "colab":
      from google.colab import ai

      self._ai = ai
      try:
        self.available = ai.list_models()
      except Exception as e:  # noqa: BLE001
        self.available = [f"(could not list models: {e})"]
      default = inspect.signature(ai.generate_text).parameters["model_name"].default
      self.model = model or default
    elif backend == "aistudio":
      from google import genai

      self._types = genai.types
      self._client = genai.Client(api_key=gemini_key())
      self.available = [m.name.removeprefix("models/") for m in self._client.models.list()
                        if "generateContent" in (m.supported_actions or [])]
      if not model:
        raise ValueError("Pass a model for AI Studio, e.g. one of: " + ", ".join(
          m for m in self.available if "flash" in m)[:600])
      self.model = model
    else:
      raise ValueError(f"unknown backend {backend!r}")

  def _once(self, system: str, user: str) -> str:
    if self.backend == "colab":
      # Colab AI takes one prompt (no system role): system text goes first.
      text = self._ai.generate_text(system + "\n\n---\n\n" + user, model_name=self.model)
    else:
      resp = self._client.models.generate_content(
        model=self.model, contents=user,
        config=self._types.GenerateContentConfig(
          system_instruction=system, response_mime_type="application/json", temperature=0.4),
      )
      text = resp.text
    if not text:
      raise RuntimeError("empty reply")
    return text

  def __call__(self, system: str, user: str, retries: int = 6):
    waited = 0.0
    for attempt in range(retries):
      t0 = perf_counter()
      try:
        return self._once(system, user), perf_counter() - t0, waited
      except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        transient = any(s in msg for s in ("429", "rate", "quota", "resource", "unavailable",
                                            "503", "500", "overloaded", "timeout", "empty reply"))
        if not transient or attempt == retries - 1:
          raise
        pause = min(60.0, 4.0 * 2**attempt)
        print(f"  [llm] {type(e).__name__}: {str(e)[:120]}... retrying in {pause:.0f} s")
        time.sleep(pause)
        waited += pause
    raise RuntimeError("unreachable")


class Stopwatch:
  """Wall-clock time per stage, and the LLM calls (minus rate-limit waits)."""

  def __init__(self, llm: LLM):
    self._llm = llm
    self.t0 = perf_counter()
    self.stages: dict[str, float] = {}
    self.waited = 0.0
    self.llm_calls: list[float] = []

  @contextlib.contextmanager
  def __call__(self, stage: str):
    s = perf_counter()
    try:
      yield
    finally:
      self.stages[stage] = self.stages.get(stage, 0.0) + perf_counter() - s

  def llm(self, system: str, user: str) -> str:
    s = perf_counter()
    text, seconds, waited = self._llm(system, user)
    self.waited += waited
    self.llm_calls.append(seconds)
    self.stages["llm"] = self.stages.get("llm", 0.0) + (perf_counter() - s - waited)
    return text

  def total(self) -> float:
    return perf_counter() - self.t0 - self.waited
