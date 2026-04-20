"""Data loading and alignment utilities."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from tow.config import get_data_root


@dataclass
class BenchmarkItem:
    """A single benchmark item with aligned fields."""

    infer_id: str
    bench_id: str
    task_type: str
    sub_task: str
    basic_instruction: str
    information: str
    reference: str
    infers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "infer_id": self.infer_id,
            "bench_id": self.bench_id,
            "task_type": self.task_type,
            "sub_task": self.sub_task,
            "basic_instruction": self.basic_instruction,
            "information": self.information,
            "reference": self.reference,
        }


def load_task_data(task: str, data_root: Path | None = None) -> list[BenchmarkItem]:
    """Load and align data for a single task (completion / guide / open).

    Returns a list of BenchmarkItem with consistent fields:
      - infer_id (unique identifier)
      - bench_id
      - basic_instruction
      - information (context for completion, outline for guide, empty for open)
      - reference
      - infers: dict mapping model_name -> generated text
    """
    if data_root is None:
        data_root = get_data_root()

    jsonl_path = data_root / task / "data.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Data file not found: {jsonl_path}")

    items: list[BenchmarkItem] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            item = BenchmarkItem(
                infer_id=raw["infer_id"],
                bench_id=raw["bench_id"],
                task_type=raw.get("task_type", task),
                sub_task=raw.get("sub_task", ""),
                basic_instruction=raw.get("basic_instruction", ""),
                information=raw.get("information", ""),
                reference=raw.get("reference", ""),
                infers=raw.get("infers", {}),
            )
            items.append(item)
    return items


def iter_all_tasks(data_root: Path | None = None) -> Iterator[tuple[str, list[BenchmarkItem]]]:
    """Iterate over all three tasks, yielding (task_name, items)."""
    if data_root is None:
        data_root = get_data_root()
    for task in ("completion", "guide", "open"):
        task_dir = data_root / task
        if task_dir.exists():
            yield task, load_task_data(task, data_root)
