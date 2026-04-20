"""Content node scoring: 5 trait-based leaf nodes under V_C."""

from tow.api import chat_completion, extract_score
from tow.config import load_prompts, render_prompt
from tow.evaluation.weight import CONTENT_TRAITS


def score_content_trait(
    trait: str,
    instruction: str,
    reference: str,
    content: str,
    settings: dict | None = None,
) -> int:
    """Score a single content trait using the corresponding prompt macro.

    Args:
        trait: One of CONTENT_TRAITS (e.g. "opening_ending")
        instruction: The writing instruction
        reference: Reference text
        content: Writing to evaluate

    Returns:
        Integer score 1-10, or 5 as fallback.
    """
    prompts = load_prompts()
    if trait not in prompts:
        raise ValueError(f"Unknown trait prompt: {trait}")

    prompt_text = render_prompt(
        prompts[trait],
        instruction=instruction,
        reference=reference,
        content=content,
    )
    response = chat_completion(prompt_text, settings)
    score = extract_score(response)
    return score if score is not None else 5


def score_all_content_traits(
    instruction: str,
    reference: str,
    content: str,
    settings: dict | None = None,
) -> dict[str, int]:
    """Score all 5 content traits.

    Returns dict mapping trait_name -> score (1-10).
    """
    scores = {}
    for trait in CONTENT_TRAITS:
        scores[trait] = score_content_trait(
            trait, instruction, reference, content, settings
        )
    return scores
