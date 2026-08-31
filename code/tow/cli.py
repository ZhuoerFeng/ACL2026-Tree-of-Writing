"""CLI entry points for Tree-of-Writing evaluation."""

import argparse
import sys


def split_data():
    """CLI: Split inference results into per-model files."""
    parser = argparse.ArgumentParser(
        description="Split embedded inference results into separate JSONL files per model."
    )
    parser.add_argument(
        "--task",
        choices=["completion", "guide", "open", "all"],
        default="all",
        help="Task to split (default: all)",
    )
    args = parser.parse_args()

    from tow.data.splitter import split_inferences, split_all_tasks

    if args.task == "all":
        split_all_tasks()
    else:
        paths = split_inferences(args.task)
        print(f"[{args.task}] Split {len(paths)} models")
        for model, path in paths.items():
            print(f"  - {model} -> {path}")


def run_tow():
    """CLI: Run Tree-of-Writing evaluation."""
    parser = argparse.ArgumentParser(
        description="Run Tree-of-Writing evaluation on benchmark data."
    )
    parser.add_argument(
        "--task",
        choices=["completion", "guide", "open"],
        required=True,
        help="Task to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to evaluate (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of items to evaluate (for testing)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to extra settings YAML file",
    )
    args = parser.parse_args()

    from tow.config import load_settings
    from tow.evaluation.pipeline import evaluate_task

    settings = load_settings(args.config)
    results = evaluate_task(
        task=args.task,
        models=args.models,
        settings=settings,
        limit=args.limit,
    )
    print(f"Completed {len(results)} evaluations.")


def run_baseline():
    """CLI: Run baseline evaluations (BLEU/ROUGE or LLM single-pass)."""
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation methods."
    )
    parser.add_argument(
        "--task",
        choices=["completion", "guide", "open"],
        required=True,
        help="Task to evaluate",
    )
    parser.add_argument(
        "--method",
        choices=["bleu_rouge", "llm"],
        required=True,
        help="Baseline method to use",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to evaluate (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items for LLM baseline (for testing)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to extra settings YAML file",
    )
    args = parser.parse_args()

    if args.method == "bleu_rouge":
        from tow.baselines.bleu_rouge import evaluate_bleu_rouge

        results = evaluate_bleu_rouge(args.task, args.models)
        for model, scores in results.items():
            print(f"\n{model}:")
            for metric, val in scores.items():
                print(f"  {metric}: {val:.4f}")
    else:
        from tow.config import load_settings
        from tow.baselines.llm_single import evaluate_llm_baseline

        settings = load_settings(args.config)
        results = evaluate_llm_baseline(
            task=args.task,
            models=args.models,
            settings=settings,
            limit=args.limit,
        )
        print(f"Completed {len(results)} LLM baseline evaluations.")


def run_infer():
    """CLI: Generate per-model writing outputs (the infers) for a task."""
    parser = argparse.ArgumentParser(
        description="Generate model writing outputs for a task (the inference step)."
    )
    parser.add_argument(
        "--task",
        choices=["completion", "guide", "open"],
        required=True,
        help="Task to generate outputs for",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id sent to the API endpoint (configured via TOW_BASE_URL)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output label / filename (default: --model). Results go to "
        "data/{task}/inferences/{name}.jsonl. Use when the API id differs from "
        "the desired file name (e.g. an openrouter:... id vs. a clean label).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=60,
        help="Number of worker threads (default: 60)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of items to generate (for testing)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to extra settings YAML file",
    )
    args = parser.parse_args()

    from tow.config import load_settings
    from tow.inference.generate import run_inference

    settings = load_settings(args.config)
    out_path = run_inference(
        task=args.task,
        model=args.model,
        name=args.name,
        parallel=args.parallel,
        limit=args.limit,
        settings=settings,
    )
    print(f"Wrote outputs to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tow.cli {split|tow|baseline|infer}")
        sys.exit(1)

    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "split":
        split_data()
    elif cmd == "tow":
        run_tow()
    elif cmd == "baseline":
        run_baseline()
    elif cmd == "infer":
        run_infer()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
