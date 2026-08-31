"""Generation loop: query a model for every item of a task and save outputs.

Preserves the original inference script's behavior:
  - retry with exponential backoff (5 attempts),
  - parse the ``<我的生成开始>...<我的生成结束>`` block from the reply,
  - a parse failure counts as a failed attempt and is retried,
  - threaded fan-out over items,
  - resume by skipping items whose ``infer_id`` is already in the output file,
  - on unrecoverable error, the item's output is saved as an empty string.

The model transport is swapped to the framework's OpenAI-compatible client
(:mod:`tow.api`), and outputs are written in the released ``inferences/`` schema
(one JSONL file per model, one record per item).
"""

import json
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from tow.api import get_client
from tow.config import get_data_root, load_settings
from tow.data.loader import BenchmarkItem
from tow.inference.prompts import build_prompt


# Default parallelism, matching the original script.
DEFAULT_PARALLEL = 60

# Extract the generated content from the required output contract.
_GEN_PATTERN = re.compile(r"<我的生成开始>(.*?)<我的生成结束>", re.DOTALL)


def extract_generation(text: str) -> str:
    """Extract the content inside the ``<我的生成开始>...<我的生成结束>`` block.

    Raises ``IndexError`` if the contract is not satisfied (which, inside
    :func:`call_api`, triggers a retry) — reproducing the original behavior.
    """
    return re.findall(_GEN_PATTERN, text)[0].strip()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=3))
def call_api(messages: list[dict], model: str, settings: dict) -> str:
    """Call the model once and return the parsed generation.

    Transport swapped to the OpenAI-compatible client. No decoding parameters
    (temperature / top_p / max_tokens) are set here: the original benchmark
    outputs were produced with each provider's defaults. Configure the endpoint
    via TOW_BASE_URL / TOW_API_KEY (see :func:`tow.config.load_settings`).
    """
    client = get_client(settings)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    output = response.choices[0].message.content or ""
    return extract_generation(output)


def _load_aligned(task: str, data_root: Path) -> list[BenchmarkItem]:
    """Load the released ``aligned.jsonl`` items for a task."""
    path = data_root / task / "aligned.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Aligned data not found: {path}")
    items: list[BenchmarkItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            items.append(
                BenchmarkItem(
                    infer_id=raw["infer_id"],
                    bench_id=raw["bench_id"],
                    task_type=raw.get("task_type", task),
                    sub_task=raw.get("sub_task", ""),
                    basic_instruction=raw.get("basic_instruction", ""),
                    information=raw.get("information", ""),
                    reference=raw.get("reference", ""),
                )
            )
    return items


def _load_done_ids(out_path: Path) -> set[str]:
    """Collect the ``infer_id``s already present in an output file (for resume)."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["infer_id"])
    return done


def run_inference(
    task: str,
    model: str,
    name: str | None = None,
    parallel: int = DEFAULT_PARALLEL,
    limit: int | None = None,
    settings: dict | None = None,
) -> Path:
    """Generate outputs for every item of *task* using *model*.

    Args:
        task: One of ``completion`` / ``guide`` / ``open``.
        model: Model id sent to the API endpoint (via TOW_BASE_URL).
        name: Output label / filename; results go to
            ``data/{task}/inferences/{name}.jsonl``. Defaults to *model* (use a
            separate name when the API id differs from the desired file name,
            e.g. an ``openrouter:...`` id vs. a clean model label).
        parallel: Number of worker threads.
        limit: Optional cap on the number of items (for quick testing).
        settings: Resolved settings dict (defaults to :func:`load_settings`).

    Returns:
        Path to the per-model output file.
    """
    if settings is None:
        settings = load_settings()
    if name is None:
        name = model

    data_root = get_data_root(settings)
    items = _load_aligned(task, data_root)
    if limit is not None:
        items = items[:limit]

    out_dir = data_root / task / "inferences"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"

    already_infers = _load_done_ids(out_path)
    write_lock = threading.Lock()

    def run_sample_and_save(item: BenchmarkItem) -> None:
        if item.infer_id in already_infers:
            return

        prompt = build_prompt(item)
        messages = [{"role": "user", "content": prompt}]

        response = ""
        try:
            response = call_api(messages, model, settings)
        except BadRequestError as e:
            print("bad request")
            print(e)
            response = ""
        except Exception as e:
            print("other error")
            print(e)
            traceback.print_exc()
            response = ""

        record = {
            "infer_id": item.infer_id,
            "bench_id": item.bench_id,
            "task_type": item.task_type,
            "sub_task": item.sub_task,
            "model": name,
            "output": response,
        }
        with write_lock:
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(parallel) as executor:
        for _ in tqdm(executor.map(run_sample_and_save, items), total=len(items)):
            pass

    return out_path
