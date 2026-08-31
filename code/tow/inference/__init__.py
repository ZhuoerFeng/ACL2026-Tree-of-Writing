"""Model inference: generate per-model writing outputs (the ``infers``).

This subpackage reproduces the generation step of HoWToBench — querying an LLM
to produce a writing ``output`` for every benchmark item of a task. The prompt
construction, the ``<我的生成开始>...<我的生成结束>`` output contract, the output
parsing, the retry policy, the threaded fan-out, and the resume-by-``infer_id``
behavior are kept exactly as in the original inference script. Only the model
transport (an OpenAI-compatible client, see :mod:`tow.api`) and the input/output
schema (the released ``aligned.jsonl`` / ``inferences/`` layout) are adapted to
fit the released framework and data.
"""

from tow.inference.prompts import build_prompt
from tow.inference.generate import run_inference, call_api, extract_generation

__all__ = ["build_prompt", "run_inference", "call_api", "extract_generation"]
