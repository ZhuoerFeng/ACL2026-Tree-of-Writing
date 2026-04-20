"""Split inference results into separate files per model."""

import json
from pathlib import Path

from tow.config import get_data_root
from tow.data.loader import load_task_data


def split_inferences(task: str, data_root: Path | None = None) -> dict[str, Path]:
    """Split embedded inference results into separate JSONL files.

    For each model found in the data, creates:
      data/{task}/inferences/{model_name}.jsonl

    Each line contains:
      {"infer_id": ..., "bench_id": ..., "model": ..., "output": ...}

    Returns a dict mapping model_name -> output file path.
    """
    if data_root is None:
        data_root = get_data_root()

    items = load_task_data(task, data_root)
    out_dir = data_root / task / "inferences"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all model names
    model_names: set[str] = set()
    for item in items:
        model_names.update(item.infers.keys())

    # Write per-model files
    result_paths: dict[str, Path] = {}
    for model in sorted(model_names):
        out_path = out_dir / f"{model}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                output_text = item.infers.get(model, "")
                record = {
                    "infer_id": item.infer_id,
                    "bench_id": item.bench_id,
                    "task_type": item.task_type,
                    "sub_task": item.sub_task,
                    "model": model,
                    "output": output_text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        result_paths[model] = out_path

    # Also write aligned base data (without infers) for convenience
    base_path = data_root / task / "aligned.jsonl"
    with open(base_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    return result_paths


def split_all_tasks(data_root: Path | None = None) -> None:
    """Split inference results for all tasks."""
    if data_root is None:
        data_root = get_data_root()
    for task in ("completion", "guide", "open"):
        task_dir = data_root / task
        if not task_dir.exists():
            print(f"Skipping {task}: directory not found")
            continue
        paths = split_inferences(task, data_root)
        print(f"[{task}] Split {len(paths)} models:")
        for model, path in paths.items():
            print(f"  - {model} -> {path}")
