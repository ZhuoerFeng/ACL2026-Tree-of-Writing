"""Score aggregation following the Tree-of-Writing methodology."""

from tow.evaluation.weight import CONTENT_TRAITS
from tow.evaluation.format_node import FORMAT_TRAITS


def aggregate_content_score(
    trait_scores: dict[str, int],
    weights: list[float],
) -> float:
    """Weighted sum of content trait scores.

    Score(V_C) = Σ w_i * Score(L_i) for content leaf nodes.
    """
    score = 0.0
    for i, trait in enumerate(CONTENT_TRAITS):
        score += weights[i] * trait_scores.get(trait, 5)
    return score


def aggregate_format_score(
    trait_scores: dict[str, int],
) -> float:
    """Average of format trait scores.

    Score(V_F) = average of format leaf node scores.
    Format traits use equal weights (1/3 each).
    """
    values = [trait_scores.get(t, 5) for t in FORMAT_TRAITS]
    return sum(values) / len(values) if values else 5.0


def aggregate_root_score(
    content_score: float,
    format_score: float,
    impression_score: float,
    task_type: str = "guide",
) -> float:
    """Aggregate V_C, V_F, V_I into root score.

    Uses averaging strategy based on number of leaf nodes:
    - Content has 5 leaf nodes
    - Format has 3 leaf nodes
    - Impression has 1 leaf node (it is itself a leaf)
    Total = 9 leaf nodes

    For completion task (no format dimension):
    - Content has 5 leaf nodes
    - Impression has 1 leaf node
    Total = 6 leaf nodes

    Score(R) = (n_C * Score(V_C) + n_F * Score(V_F) + n_I * Score(V_I)) / (n_C + n_F + n_I)
    """
    if task_type == "completion":
        # No format dimension for completion task
        n_c, n_f, n_i = 5, 0, 1
    else:
        n_c, n_f, n_i = 5, 3, 1

    total_leaves = n_c + n_f + n_i
    score = (
        n_c * content_score + n_f * format_score + n_i * impression_score
    ) / total_leaves
    return round(score, 4)
