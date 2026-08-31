"""Prompt construction for the inference (generation) step.

The prompt strings and the ``<我的生成开始>...<我的生成结束>`` output contract are
reproduced verbatim from the original inference script — do not edit the wording
or formatting, as the released benchmark outputs were generated with these exact
prompts.

The only change from the original is the ``guide`` branch, which reads the
grounding text from ``information`` (the field name in the released data) instead
of the original script's internal ``revision`` field. The two hold the same
content; the field was simply renamed on release.
"""

from tow.data.loader import BenchmarkItem


def build_prompt(item: BenchmarkItem) -> str:
    """Build the generation prompt for a single benchmark item.

    Reproduces the exact per-task prompt construction of the original script.
    """
    if item.task_type == "completion":
        prompt = item.basic_instruction
        prompt += "[待填充文本开始]\n" + item.information + "\n[待填充文本结束]\n"
        prompt += "请开始填充生成任务。在以下文本中，需在 [fill in the blanks] 位置填入合理的内容，用以替换 [fill in the blanks]。请务必以“<我的生成开始>生成内容<我的生成结束>”的格式进行填充。\n如果你需要对填充内容进行解释，请在填充内容之后，用“<解释开始>解释内容<解释结束>”的格式进行解释。"

    elif item.task_type == "guide":
        prompt = item.basic_instruction
        # Original script read this from ``input_item['revision']``; the released
        # data stores the same grounding text under ``information``.
        prompt += "\n\n下面是你写作过程中的必要信息\n\n" + item.information
        prompt += "\n请务必以“<我的生成开始>生成内容<我的生成结束>”的格式生成你的写作。"

    elif item.task_type == "open":
        prompt = item.basic_instruction
        prompt += "\n请务必以“<我的生成开始>生成内容<我的生成结束>”的格式生成你的写作。"

    else:
        raise ValueError("task type not supported")

    return prompt
