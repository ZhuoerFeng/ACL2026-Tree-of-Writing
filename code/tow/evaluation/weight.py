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
    weight_sum = sum(weights)
    if len(weights) == 4 and all(-1.0 <= w <= 1.0 for w in weights) and weight_sum > 1e-9:
        # Insert proper_instance weight (index 2) as average of others, then renormalize
        avg = weight_sum / len(weights)
        proper_w = max(0.0, min(avg, 0.3))
        weights_5 = [weights[0], weights[1], proper_w, weights[2], weights[3]]
        total = sum(weights_5)
        return [w / total for w in weights_5]

    # Fallback: equal weights
    return [0.2] * 5


def _parse_weights(response: str) -> list[float]:
    """Parse weight values from LLM response."""
    json_start = response.find("{")
    if json_start != -1:
        try:
            data, _ = json.JSONDecoder().raw_decode(response, json_start)
            values = data["weights"]
            if isinstance(values, list) and not any(isinstance(w, bool) for w in values):
                return [float(w) for w in values]
        except (ValueError, KeyError, TypeError, OverflowError):
            pass

        # A failed JSON attempt may contain unrelated numbers in its explanation.
        return []

    # Fallback for plain text responses
    numbers = re.findall(r"[-+]?\d*\.?\d+", response)
    floats = [float(n) for n in numbers if -1.0 <= float(n) <= 1.0]
    if len(floats) >= 4:
        return floats[:4]

    return []
