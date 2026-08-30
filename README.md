# HoWToBench & Tree-of-Writing (ToW)

**Ho**listic Evaluation for LLM's Capability in Human-level **W**riting using **T**ree **o**f Writing.

[![Paper](https://img.shields.io/badge/ACL_2026-2026.acl--long.317-b31b1b.svg)](https://aclanthology.org/2026.acl-long.317/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.19071-b31b1b.svg)](https://arxiv.org/abs/2604.19071)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HowToBench-yellow.svg)](https://huggingface.co/datasets/andrewzhuoerfeng/HowToBench)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the **benchmark data**, **evaluation code**, and **paper** for HoWToBench, a large-scale Chinese writing benchmark, together with the **Tree-of-Writing (ToW)** evaluation method.

Evaluating LLM writing is hard: writing skill is multidimensional, and naive LLM-as-a-judge averaging of sub-features is inconsistent and opaque (*Negotiation Inconsistency*). **ToW** scores writing on a structured tree whose leaves are individual traits and whose aggregation weights are set per-instruction by an LLM negotiator — reaching **0.93 Pearson correlation** with human rankings and staying robust to textual disturbances that fool other metrics. **HoWToBench** provides **1,302** expert-curated instructions across **12 genres** and **3 task formats**, each with a high-quality human reference and outputs from **10 flagship LLMs**.

## Dataset

The benchmark data is hosted on the Hugging Face Hub:
👉 **https://huggingface.co/datasets/andrewzhuoerfeng/HowToBench**

```python
from datasets import load_dataset

# Tasks: "completion", "guide", "open"
ds = load_dataset("andrewzhuoerfeng/HowToBench", "guide", split="benchmark")
print(ds[0]["basic_instruction"], ds[0]["reference"])
```

Or download the raw files (used by the code below):

```bash
hf download andrewzhuoerfeng/HowToBench --repo-type dataset --local-dir data
```

| Level | Task | Items | Description |
|-------|------|-------|-------------|
| I | `completion` | 379 | Fill in `[fill in the blank]` spans omitted from a human text. |
| II | `guide` | 456 | Expand a given outline / guidance into a full piece. |
| III | `open` | 467 | Write from a one-sentence topic and genre alone. |

See the [dataset card](https://huggingface.co/datasets/andrewzhuoerfeng/HowToBench) for genres, fields, and construction details.

## Project Structure

```
ACL2026-ToW/
├── data/                      # Benchmark data (mirrors the HF dataset)
│   ├── completion/
│   │   ├── aligned.jsonl      # Benchmark items (instruction / grounding / reference)
│   │   └── inferences/        # Per-model outputs, one JSONL per model
│   ├── guide/                 # Guided writing task data
│   └── open/                  # Open writing task data
└── code/                      # Source code
    ├── pyproject.toml
    ├── requirements.txt
    ├── config/
    │   ├── settings.yaml      # API and evaluation settings
    │   └── prompts.yaml       # All prompt macros
    └── tow/                   # Python package
        ├── config.py          # Configuration loader
        ├── api.py             # OpenAI-compatible API client
        ├── cli.py             # CLI entry points
        ├── data/
        │   ├── loader.py      # Data loading and alignment
        │   └── splitter.py    # Inference result splitting
        ├── evaluation/
        │   ├── pipeline.py    # Main ToW evaluation pipeline
        │   ├── weight.py      # Edge weighting (LLM-based)
        │   ├── content.py     # Content trait scoring (5 traits)
        │   ├── format_node.py # Format scoring (LLM + regex)
        │   ├── impression.py  # Impression scoring
        │   └── aggregator.py  # Tree score aggregation
        └── baselines/
            ├── bleu_rouge.py  # BLEU/ROUGE metrics
            └── llm_single.py  # Single-pass LLM evaluation
```

## Installation

```bash
cd code
pip install -e .
```

## Configuration

### API Settings

Set your API credentials via environment variables:

```bash
export TOW_API_KEY="your-api-key"
export TOW_BASE_URL="https://api.openai.com/v1"  # or compatible endpoint
export TOW_MODEL="gpt-4o"
```

Or edit `code/config/settings.yaml` directly.

### Prompt Macros

All evaluation prompts are defined in `code/config/prompts.yaml`. You can modify them to experiment with different evaluation strategies.

## Usage

### 1. Run Tree-of-Writing Evaluation

```bash
tow-eval --task guide                             # Evaluate all models on guide task
tow-eval --task open --models gpt-4o deepseek-r1  # Specific models
tow-eval --task completion --limit 10             # Test with 10 items
```

### 2. Run Baselines

```bash
# BLEU/ROUGE (no API needed)
tow-baseline --task guide --method bleu_rouge

# Single-pass LLM evaluation
tow-baseline --task guide --method llm --limit 10
```

## Tree-of-Writing Methodology

The evaluation uses a hierarchical tree structure:

```
Root (Final Score)
├── Content (V_C) — weighted sum of 5 traits
│   ├── Opening & Ending        (1-10, LLM-scored)
│   ├── Language & Rhetoric      (1-10, LLM-scored)
│   ├── Proper Instance          (1-10, LLM-scored)
│   ├── Argumentative Logic      (1-10, LLM-scored)
│   └── Emotion                  (1-10, LLM-scored)
├── Format (V_F) — average of 3 traits
│   ├── Plots & Structure        (0/5/10, LLM-scored)
│   ├── Formatting               (0/5/10, regex-based)
│   └── Paragraphing             (0/5/10, LLM-scored)
└── Impression (V_I)             (1-10, LLM-scored)
```

**Key steps:**
1. **Edge Weighting**: an LLM negotiator assigns weights to content traits based on the writing instruction (genre, task type, requirements).
2. **Trait Scoring**: each leaf node is scored independently using trait-specific expert prompts.
3. **Aggregation**: scores are aggregated bottom-up through weighted sums for a transparent, reproducible final score.

## Data Format

Each benchmark item (`data/<task>/aligned.jsonl`) contains:
- `infer_id`: Unique identifier
- `bench_id`: Benchmark identifier
- `basic_instruction`: Writing instruction (`I`)
- `information`: Grounding (`G`) — context (completion) / outline (guide) / empty (open)
- `reference`: Human ground-truth reference text (`R`)
- `task_type`: completion / guide / open
- `sub_task`: Genre (e.g., 议论文, 小说)

Each model's outputs live in `data/<task>/inferences/<model>.jsonl` with fields `infer_id`, `bench_id`, `task_type`, `sub_task`, `model`, and `output` (the generated writing `W`). Join to items on `infer_id`.

## Models Evaluated

10 flagship LLMs are included in the benchmark:
- claude-3-5-sonnet-20241022, claude-3-haiku-20240307
- deepseek-chat (V3), deepseek-r1
- doubao-pro-32k-241215
- glm-4-plus-250111
- gemini-2.0-flash
- gpt-4o-2024-11-20
- Llama-3.3-70B-Instruct
- o3-mini-2025-01-31

## License

Code in this repository is released under the [Apache-2.0 License](LICENSE). The benchmark data on Hugging Face is released under CC-BY-4.0; model outputs are provided for research purposes and remain subject to their providers' terms.

## Citation

If you use HoWToBench or Tree-of-Writing in your research, please cite:

```bibtex
@inproceedings{feng-etal-2026-howtobench,
    title = "{H}o{WT}o{B}ench: Holistic Evaluation for {LLM}{'}s Capability in Human-level Writing using Tree of Writing",
    author = "Feng, Andrew Zhuoer  and
      Wang, Cunxiang  and
      Luo, Yu  and
      Fan, Lin  and
      Zhou, Irene  and
      Wang, Zikang  and
      Gu, Xiaotao  and
      Tang, Jie  and
      Wang, Hongning  and
      Huang, Minlie",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.317/",
    doi = "10.18653/v1/2026.acl-long.317",
    pages = "6986--7034",
    ISBN = "979-8-89176-390-6",
}
```
