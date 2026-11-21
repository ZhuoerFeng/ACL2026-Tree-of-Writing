import math

import pytest

from tow.data.loader import BenchmarkItem
from tow.evaluation import pipeline, weight


@pytest.mark.parametrize(
    "response",
    [
        '{"weights": [0, 0, 0, 0]}',
        '{"weights": [1, -1, 0, 0]}',
        '{"weights": [1, -0.9999999999, 0, 0]}',
        '{"weights": [-0.8, 0.1, 0.2, 0.1]}',
        '{"weights": [NaN, 0.2, 0.3, 0.5]}',
        '{"weights": [Infinity, 0.2, 0.3, 0.5]}',
        '{"weights": [-Infinity, 0.2, 0.3, 0.5]}',
        '{"weights": [1.1, 0.2, 0.3, 0.4]}',
        '{"weights": [-1.1, 0.8, 0.7, 0.6]}',
        '{"weights": ["n/a", 0.2, 0.3, 0.5]}',
        '{"weights": ["n/a", 0.2, 0.3, 0.5], "explanation": "confidence 0.9"}',
        pytest.param('{"weights": [' + str(10**309) + ', 0.2, 0.3, 0.5]}', id="overflow"),
        '{"scores": [0.4, 0.3, 0.2, 0.1]}',
        '{"weights": [0.4, 0.3, 0.2]}',
        '{"weights": [0.4, 0.3, 0.2, 0.1, 0.5]}',
        '{"weights": [0.4, 0.3, 0.2, 0.1,]}',
        '{"weights": [0.4, 0.3, 0.2, 0.1]',
        '{"weights": {"0.4": 1, "0.3": 2, "0.2": 3, "0.1": 4}}',
        '{"weights": [true, 0, 0, 0]}',
    ],
)
def test_invalid_weights_use_equal_fallback(monkeypatch, response):
    monkeypatch.setattr(weight, "chat_completion", lambda *_: response)

    weights = weight.plan_content_weights("Write an essay.")

    assert weights == [0.2, 0.2, 0.2, 0.2, 0.2]
    assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"weights": [0.4, 0.3, 0.2, 0.1]}', [0.32, 0.24, 0.2, 0.16, 0.08]),
        ('{"weights": [-1, 1, 0.5, 0.5]}', [-0.8, 0.8, 0.2, 0.4, 0.4]),
        ('{"weights": ["0.2", "0.4", "0", "0.4"]}', [0.16, 0.32, 0.2, 0, 0.32]),
        (
            'Here are the weights.\n```json\n{"weights": [0.4, 0.3, 0.2, 0.1]}\n```\nDone.',
            [0.32, 0.24, 0.2, 0.16, 0.08],
        ),
        (
            '{"weights": [0.4, 0.3, 0.2, 0.1], "explanation": "A literal } in the text."}',
            [0.32, 0.24, 0.2, 0.16, 0.08],
        ),
        ("Weights: -0.2, 0.4, 0.4, 0.4", [-0.16, 0.32, 0.2, 0.32, 0.32]),
        ('{"weights": [0.2, 0.1, 0.1, 0.1]}', [0.32, 0.16, 0.2, 0.16, 0.16]),
        ('{"weights": [0.8, 0.6, 0.4, 0.2]}', [8 / 23, 6 / 23, 3 / 23, 4 / 23, 2 / 23]),
    ],
)
def test_valid_weights_preserve_five_trait_normalization(monkeypatch, response, expected):
    monkeypatch.setattr(weight, "chat_completion", lambda *_: response)

    weights = weight.plan_content_weights("Write an essay.")

    assert weights == pytest.approx(expected)
    assert all(math.isfinite(value) for value in weights)
    assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("response", "task_type", "expected_content", "expected_root"),
    [
        ('{"weights": [0.4, 0.3, 0.2, 0.1]}', "guide", 4.88, 5.1556),
        ('{"weights": [-0.2, 0.4, 0.4, 0.4]}', "guide", 7.92, 6.8444),
        ('{"weights": [0, 0, 0, 0]}', "guide", 6.0, 5.7778),
        ('{"weights": [NaN, 0.2, 0.3, 0.5]}', "completion", 6.0, 6.1667),
        ('{"weights": ["n/a", 0.2, 0.3, 0.5]}', "guide", 6.0, 5.7778),
    ],
)
def test_weights_feed_content_and_root_scores(
    monkeypatch, response, task_type, expected_content, expected_root
):
    monkeypatch.setattr(weight, "chat_completion", lambda *_: response)
    monkeypatch.setattr(
        pipeline,
        "score_all_content_traits",
        lambda *_: {
            "opening_ending": 2,
            "language_rhetoric": 4,
            "proper_instance": 6,
            "argumentative_logic": 8,
            "emotion": 10,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "score_all_format_traits",
        lambda *_: {"plots_structure": 0, "formatting": 5, "paragraphing": 10},
    )
    monkeypatch.setattr(pipeline, "score_impression", lambda *_: 7)
    item = BenchmarkItem(
        infer_id="fixture",
        bench_id="fixture",
        task_type=task_type,
        sub_task="essay",
        basic_instruction="Write an essay.",
        information="",
        reference="Reference essay.",
        infers={"model": "Generated essay."},
    )

    result = pipeline.evaluate_single(item, "model")

    assert result.content_score == pytest.approx(expected_content)
    assert result.root_score == pytest.approx(expected_root)
