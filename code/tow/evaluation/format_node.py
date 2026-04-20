"""Format node scoring: 3 leaf nodes under V_F (plots, formatting, paragraphing)."""

import re

from tow.api import chat_completion, extract_score
from tow.config import load_prompts, render_prompt


FORMAT_TRAITS = ["plots_structure", "formatting", "paragraphing"]


def score_plots_structure(
    instruction: str,
    reference: str,
    content: str,
    settings: dict | None = None,
) -> int:
    """LLM-based scoring for plot design and structure (step: 0, 5, 10)."""
    prompts = load_prompts()
    prompt_text = render_prompt(
        prompts["plots_structure"],
        instruction=instruction,
        reference=reference,
        content=content,
    )
    response = chat_completion(prompt_text, settings)
    raw_score = extract_score(response)
    if raw_score is None:
        return 5
    # Map 1-10 to step function: {0, 5, 10}
    return _to_step(raw_score)


def score_formatting(content: str) -> int:
    """Regex-based scoring for title formatting and hierarchy.

    Checks:
    1. Title detection (markdown headers, Chinese-style numbered titles)
    2. Hierarchical consistency
    3. List formatting

    Returns: 0, 5, or 10
    """
    lines = content.strip().split("\n")

    # Detect titles: markdown headers or Chinese-style headers
    md_header_pattern = re.compile(r"^#{1,6}\s+.+")
    cn_header_pattern = re.compile(
        r"^[一二三四五六七八九十百]+[、.．]\s*.+"
        r"|^第[一二三四五六七八九十百]+[章节部分]\s*.+"
        r"|^\d+[、.．]\s*.+"
    )

    titles = []
    has_list = False
    list_in_prose = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if md_header_pattern.match(stripped):
            level = len(stripped) - len(stripped.lstrip("#"))
            titles.append(("md", level, stripped))
        elif cn_header_pattern.match(stripped):
            titles.append(("cn", 1, stripped))

        # Check for list markers in continuous text
        if re.match(r"^[-*•]\s+", stripped) or re.match(r"^\d+[.)]\s+", stripped):
            has_list = True

    # Scoring logic
    if not titles:
        # No titles found - might be acceptable for short texts
        if len(content) < 500:
            return 10  # Short text, no titles needed
        return 5  # Moderate: no titling in longer text

    # Check hierarchical consistency
    hierarchy_ok = True
    if len(titles) >= 2:
        levels = [t[1] for t in titles if t[0] == "md"]
        if levels:
            # Check if levels are monotonically structured
            for i in range(1, len(levels)):
                if levels[i] - levels[i - 1] > 1:
                    hierarchy_ok = False
                    break

    if not hierarchy_ok:
        return 0  # Violations in hierarchy
    if has_list and len(titles) == 0:
        return 5  # Lists without structure
    return 10


def score_paragraphing(
    instruction: str,
    content: str,
    settings: dict | None = None,
) -> int:
    """LLM-based scoring for paragraph and chapter division (raw: 1-3, mapped to 0/5/10)."""
    prompts = load_prompts()
    prompt_text = render_prompt(
        prompts["paragraphing"],
        instruction=instruction,
        content=content,
    )
    response = chat_completion(prompt_text, settings)
    raw_score = extract_score(response)
    if raw_score is None:
        return 5
    # Paper specifies: y = 5 * (x - 1) for mapping 1-3 to 0-10
    mapped = 5 * (min(max(raw_score, 1), 3) - 1)
    return mapped


def score_all_format_traits(
    instruction: str,
    reference: str,
    content: str,
    settings: dict | None = None,
) -> dict[str, int]:
    """Score all 3 format traits.

    Returns dict mapping trait_name -> score (0, 5, or 10).
    """
    return {
        "plots_structure": score_plots_structure(
            instruction, reference, content, settings
        ),
        "formatting": score_formatting(content),
        "paragraphing": score_paragraphing(instruction, content, settings),
    }


def _to_step(score: int) -> int:
    """Map 1-10 score to step function {0, 5, 10}."""
    if score <= 3:
        return 0
    elif score <= 7:
        return 5
    else:
        return 10
