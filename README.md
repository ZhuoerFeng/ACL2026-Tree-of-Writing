# Tree-of-Writing (ToW) Evaluation System

Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing.

## Project Structure

```
ACL2026-ToW/
├── README.md                  # This file
├── paper/                     # LaTeX paper source
├── data/                      # Benchmark data
│   ├── completion/            # Completion task data
│   │   ├── data.jsonl         # Raw data with embedded inferences
│   │   ├── aligned.jsonl      # Aligned base data (after split)
│   │   └── inferences/        # Per-model inference files (after split)
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

### 1. Split Data

Split the embedded inference results into separate per-model files:

```bash
tow-split                    # Split all tasks
tow-split --task completion   # Split a specific task
```

### 2. Run Tree-of-Writing Evaluation

```bash
tow-eval --task guide                          # Evaluate all models on guide task
tow-eval --task open --models gpt-4o deepseek-r1  # Specific models
tow-eval --task completion --limit 10           # Test with 10 items
```

### 3. Run Baselines

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
│   ├── Opening & Ending       (1-10, LLM-scored)
│   ├── Language & Rhetoric     (1-10, LLM-scored)
│   ├── Proper Instance         (1-10, LLM-scored)
│   ├── Argumentative Logic     (1-10, LLM-scored)
│   └── Emotion                 (1-10, LLM-scored)
├── Format (V_F) — average of 3 traits
│   ├── Plots & Structure       (0/5/10, LLM-scored)
│   ├── Formatting              (0/5/10, regex-based)
│   └── Paragraphing            (0/5/10, LLM-scored)
└── Impression (V_I)            (1-10, LLM-scored)
```

**Key steps:**
1. **Edge Weighting**: LLM assigns weights to content traits based on the writing instruction
2. **Trait Scoring**: Each leaf node is scored independently using trait-specific prompts
3. **Aggregation**: Scores are aggregated bottom-up through weighted sums

## Data Format

Each JSONL record contains:
- `infer_id`: Unique identifier
- `bench_id`: Benchmark identifier
- `basic_instruction`: Writing task instruction
- `information`: Context (completion) / guided outline (guide) / empty (open)
- `reference`: Ground truth reference text
- `task_type`: completion / guide / open
- `sub_task`: Genre (e.g., 议论文, 小说)
- `infers`: Dict mapping model names to generated text

## Models Evaluated

10 LLMs are included in the benchmark:
- claude-3-5-sonnet-20241022, claude-3-haiku-20240307
- deepseek-chat, deepseek-r1
- doubao-pro-32k-241215
- glm-4-plus-250111
- gemini-2.0-flash
- gpt-4o-2024-11-20
- Llama-3.3-70B-Instruct
- o3-mini-2025-01-31
