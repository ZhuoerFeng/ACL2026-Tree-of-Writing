"""Tree-of-Writing evaluation pipeline."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from tow.data.loader import BenchmarkItem, load_task_data
from tow.config import get_data_root
from tow.evaluation.weight import plan_content_weights
from tow.evaluation.content import score_all_content_traits
from tow.evaluation.format_node import score_all_format_traits
from tow.evaluation.impression import score_impression
from tow.evaluation.aggregator import (
    aggregate_content_score,
    aggregate_format_score,
    aggregate_root_score,
)


@dataclass
class ToWResult:
    """Result of a Tree-of-Writing evaluation."""

    infer_id: str
    bench_id: str
    model: str
    task_type: str

    # Weights
    content_weights: list[float]

    # Leaf scores
    content_trait_scores: dict[str, int]
    format_trait_scores: dict[str, int]
    impression_score: int

    # Aggregated scores
    content_score: float
    format_score: float
    root_score: float


def evaluate_single(
    item: BenchmarkItem,
    model: str,
    settings: dict | None = None,
) -> ToWResult:
    """Run full ToW evaluation on a single item for a specific model.

    Steps (following the methodology):
    1. Plan content weights via LLM
    2. Score content traits (5 leaf nodes)
    3. Score format traits (3 leaf nodes, hybrid LLM + regex)
    4. Score impression (1 leaf node)
    5. Aggregate scores up the tree
    """
    content_text = item.infers.get(model, "")
    if not content_text:
        raise ValueError(f"No inference found for model '{model}' in {item.infer_id}")

    instruction = item.basic_instruction
    if item.information:
        instruction = f"{instruction}\n\n{item.information}"
    reference = item.reference

    # Step 1: Plan weights
    content_weights = plan_content_weights(instruction, settings)

    # Step 2: Score content traits
    content_traits = score_all_content_traits(
        instruction, reference, content_text, settings
    )

    # Step 3: Score format traits
    if item.task_type == "completion":
        format_traits = {"plots_structure": 5, "formatting": 5, "paragraphing": 5}
    else:
        format_traits = score_all_format_traits(
            instruction, reference, content_text, settings
        )

    # Step 4: Score impression
    imp_score = score_impression(instruction, reference, content_text, settings)

    # Step 5: Aggregate
    c_score = aggregate_content_score(content_traits, content_weights)
    f_score = aggregate_format_score(format_traits)
    r_score = aggregate_root_score(c_score, f_score, imp_score, item.task_type)

    return ToWResult(
        infer_id=item.infer_id,
        bench_id=item.bench_id,
        model=model,
        task_type=item.task_type,
        content_weights=content_weights,
        content_trait_scores=content_traits,
        format_trait_scores=format_traits,
        impression_score=imp_score,
        content_score=round(c_score, 4),
        format_score=round(f_score, 4),
        root_score=r_score,
    )


def evaluate_task(
    task: str,
    models: list[str] | None = None,
    output_dir: Path | None = None,
    settings: dict | None = None,
    limit: int | None = None,
) -> list[ToWResult]:
    """Run ToW evaluation for an entire task.

    Args:
        task: "completion", "guide", or "open"
        models: List of model names to evaluate. None = all models.
        output_dir: Where to save results. Defaults to data/{task}/results/
        settings: API settings dict.
        limit: Max number of items to evaluate (for testing).

    Returns:
        List of ToWResult objects.
    """
    data_root = get_data_root(settings)
    items = load_task_data(task, data_root)
    if limit:
        items = items[:limit]

    if output_dir is None:
        output_dir = data_root / task / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models
    if models is None:
        all_models: set[str] = set()
        for item in items:
            all_models.update(item.infers.keys())
        models = sorted(all_models)

    results: list[ToWResult] = []
    for model in models:
        out_path = output_dir / f"tow_{model}.jsonl"
        print(f"Evaluating {task}/{model} ...")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in tqdm(items, desc=f"{model}"):
                if model not in item.infers:
                    continue
                try:
                    result = evaluate_single(item, model, settings)
                    results.append(result)
                    f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"  Error on {item.infer_id}: {e}")

    return results
