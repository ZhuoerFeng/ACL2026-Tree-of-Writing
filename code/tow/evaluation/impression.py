"""Impression node scoring (V_I): leaf node directly under root."""

from tow.api import chat_completion, extract_score
from tow.config import load_prompts, render_prompt


def score_impression(
    instruction: str,
    reference: str,
    content: str,
    settings: dict | None = None,
) -> int:
    """Score overall impression (1-10)."""
    prompts = load_prompts()
    prompt_text = render_prompt(
        prompts["impression"],
        instruction=instruction,
        reference=reference,
        content=content,
    )
    response = chat_completion(prompt_text, settings)
    score = extract_score(response)
    return score if score is not None else 5
