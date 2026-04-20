"""BLEU and ROUGE baseline metrics for writing evaluation."""

import json
from pathlib import Path

import jieba
from rouge_chinese import Rouge

from tow.data.loader import BenchmarkItem, load_task_data
from tow.config import get_data_root


def _tokenize_chinese(text: str) -> str:
    """Tokenize Chinese text using jieba for BLEU/ROUGE computation."""
    return " ".join(jieba.cut(text))


def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F-scores for Chinese text."""
    hyp_tok = _tokenize_chinese(hypothesis)
    ref_tok = _tokenize_chinese(reference)

    if not hyp_tok.strip() or not ref_tok.strip():
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    rouge = Rouge()
    try:
        scores = rouge.get_scores(hyp_tok, ref_tok)[0]
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
    except Exception:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}


def compute_bleu(hypothesis: str, reference: str, max_n: int = 4) -> dict[str, float]:
    """Compute BLEU-1 to BLEU-N for Chinese text using simple n-gram overlap."""
    hyp_tokens = list(jieba.cut(hypothesis))
    ref_tokens = list(jieba.cut(reference))

    if not hyp_tokens or not ref_tokens:
        return {f"bleu-{i}": 0.0 for i in range(1, max_n + 1)}

    results = {}
    for n in range(1, max_n + 1):
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not hyp_ngrams:
            results[f"bleu-{n}"] = 0.0
            continue

        # Count matches
        matches = 0
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1

        for ng in hyp_ngrams:
            if ref_counts.get(ng, 0) > 0:
                matches += 1
                ref_counts[ng] -= 1

        precision = matches / len(hyp_ngrams)
        results[f"bleu-{n}"] = round(precision, 6)

    return results


def _get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def evaluate_bleu_rouge(
    task: str,
    models: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Run BLEU/ROUGE evaluation for a task.

    Returns dict mapping model -> averaged metric scores.
    """
    data_root = get_data_root()
    items = load_task_data(task, data_root)

    if output_dir is None:
        output_dir = data_root / task / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models
    if models is None:
        all_models: set[str] = set()
        for item in items:
            all_models.update(item.infers.keys())
        models = sorted(all_models)

    model_results: dict[str, dict[str, float]] = {}

    for model in models:
        all_scores: list[dict[str, float]] = []
        for item in items:
            text = item.infers.get(model, "")
            if not text or not item.reference:
                continue
            bleu = compute_bleu(text, item.reference)
            rouge = compute_rouge(text, item.reference)
            combined = {**bleu, **rouge}
            all_scores.append(combined)

        # Average
        if all_scores:
            avg: dict[str, float] = {}
            for key in all_scores[0]:
                avg[key] = round(
                    sum(s[key] for s in all_scores) / len(all_scores), 6
                )
            model_results[model] = avg
        else:
            model_results[model] = {}

    # Save
    out_path = output_dir / "bleu_rouge.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False, indent=2)
    print(f"BLEU/ROUGE results saved to {out_path}")

    return model_results
