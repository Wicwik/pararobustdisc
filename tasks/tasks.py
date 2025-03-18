from collections import OrderedDict

from tasks.mmlu import MMLU, MMLUParaphrases

from transformers import PreTrainedTokenizer

TASK_MAPPING = OrderedDict(
    [
        ("mmlu", MMLU),
        ("mmlu_paraphrases", MMLUParaphrases),
    ]
)

class AutoTask:
    @classmethod
    def get(self, task : str, tokenizer : PreTrainedTokenizer, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](tokenizer, seed=seed)

        raise ValueError(
            f"Unrecognized task {task} for AutoTask.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )





