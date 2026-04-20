"""Single-pass LLM evaluation baseline."""

import json
from pathlib import Path

from tqdm import tqdm

from tow.api import chat_completion, extract_score
from tow.config import load_prompts, render_prompt, get_data_root
from tow.data.loader import BenchmarkItem, load_task_data


def evaluate_single_pass(
    item: BenchmarkItem,
    model: str,
    settings: dict | None = None,
) -> dict:
    """Evaluate a single item with a one-shot LLM call.

    Uses the llm_single_pass prompt macro.
    """
    prompts = load_prompts()
    content_text = item.infers.get(model, "")
    if not content_text:
        return {"infer_id": item.infer_id, "model": model, "score": None}

    instruction = item.basic_instruction
    if item.information:
        instruction = f"{instruction}\n\n{item.information}"

    prompt_text = render_prompt(
        prompts["llm_single_pass"],
        instruction=instruction,
        reference=item.reference,
        content=content_text,
    )
    response = chat_completion(prompt_text, settings)
    score = extract_score(response)

    return {
        "infer_id": item.infer_id,
        "bench_id": item.bench_id,
        "model": model,
        "task_type": item.task_type,
        "score": score,
        "response": response,
    }


def evaluate_llm_baseline(
    task: str,
    models: list[str] | None = None,
    output_dir: Path | None = None,
    settings: dict | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Run single-pass LLM evaluation for a task.

    Args:
        task: "completion", "guide", or "open"
        models: List of model names to evaluate.
        output_dir: Where to save results.
        settings: API settings.
        limit: Max items to evaluate.

    Returns:
        List of result dicts.
    """
    data_root = get_data_root(settings)
    items = load_task_data(task, data_root)
    if limit:
        items = items[:limit]

    if output_dir is None:
        output_dir = data_root / task / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        all_models: set[str] = set()
        for item in items:
            all_models.update(item.infers.keys())
        models = sorted(all_models)

    results: list[dict] = []
    for model in models:
        out_path = output_dir / f"llm_baseline_{model}.jsonl"
        print(f"LLM baseline: {task}/{model} ...")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in tqdm(items, desc=f"{model}"):
                if model not in item.infers:
                    continue
                try:
                    result = evaluate_single_pass(item, model, settings)
                    results.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"  Error on {item.infer_id}: {e}")

    return results
