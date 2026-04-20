"""Edge weighting: LLM-based weight assignment for content traits."""

import json
import re

from tow.api import chat_completion
from tow.config import load_prompts, render_prompt


# Content trait names in order
CONTENT_TRAITS = [
    "opening_ending",
    "language_rhetoric",
    "proper_instance",
    "argumentative_logic",
    "emotion",
]

# Trait display names matching the edge-weighting prompt dimensions
TRAIT_LABELS = [
    "Introduction and Conclusion",
    "Language and Rhetoric",
    "Proper Instance",
    "Argumentative Logic",
    "Emotional Expression",
]


def plan_content_weights(
    instruction: str, settings: dict | None = None
) -> list[float]:
    """Use LLM to determine weights for the 4 content traits.

    The edge-weighting prompt asks for 4 weights (opening, language, logic, emotion).
    We insert a default weight for "proper_instance" and renormalize.

    Returns a list of 5 weights corresponding to CONTENT_TRAITS.
    """
    prompts = load_prompts()
    prompt_text = render_prompt(prompts["edge_weighting"], instruction=instruction)
    response = chat_completion(prompt_text, settings)

    # Parse JSON from response
    weights = _parse_weights(response)

    # The prompt only asks for 4 dimensions.
    # We assign a default to "proper_instance" and renormalize to 5 dims.
    if len(weights) == 4:
        # Insert proper_instance weight (index 2) as average of others, then renormalize
        avg = sum(weights) / len(weights)
        proper_w = max(0.0, min(avg, 0.3))
        weights_5 = [weights[0], weights[1], proper_w, weights[2], weights[3]]
        total = sum(weights_5)
        if abs(total) > 1e-9:
            weights_5 = [w / total for w in weights_5]
        return weights_5

    # Fallback: equal weights
    return [0.2] * 5


def _parse_weights(response: str) -> list[float]:
    """Parse weight values from LLM response."""
    # Try JSON parse
    try:
        # Find JSON object in response
        json_match = re.search(r"\{.*?\"weights\".*?\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return [float(w) for w in data["weights"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fallback: extract numbers
    numbers = re.findall(r"[-+]?\d*\.?\d+", response)
    floats = [float(n) for n in numbers if -1.0 <= float(n) <= 1.0]
    if len(floats) >= 4:
        return floats[:4]

    return [0.25, 0.25, 0.25, 0.25]
