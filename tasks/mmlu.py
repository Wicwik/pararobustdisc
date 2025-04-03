import datasets

from datasets import Dataset

from metrics import (
    exact_match,
    macro_f1,
)

from tasks.abstract import AbstractTask


class MMLU(AbstractTask):
    name = "mmlu"
    labels_list = ["A", "B", "C", "D"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "train",
        "test": "test",
    }
    label_column_name = "answer"
    id2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    def load_dataset(self, split, subset=None) -> Dataset:
        return datasets.load_dataset("rbelanec/mmlu_paraphrases", subset, split=split)

    def preprocessor(self, example):
        input_texts = [
            "Select the best answer from the given options. Respond only with the letter corresponding to the correct choice.",
            example["input_text"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.apply_template(
            {
                "content": "\n".join(input_texts) + "\n",
                "target": " ".join(label_texts),
                "role": "user",
            }
        )

class MMLUParaphrases(AbstractTask):
    name = "mmlu_paraphrases"
    labels_list = ["A", "B", "C", "D"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "train",
        "test": "test",
    }
    label_column_name = "answer"
    id2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    def load_dataset(self, split, subset=None) -> Dataset:
        return datasets.load_dataset("rbelanec/mmlu_paraphrases", subset, split=split)

    def preprocessor(self, example):
        input_texts = [
            "Select the best answer from the given options. Respond only with the letter corresponding to the correct choice.",
            example["input_text"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.apply_template(
            {
                "content": "\n".join(input_texts) + "\n",
                "target": " ".join(label_texts),
                "role": "user",
            }
        )

class MMLUAdversarial(AbstractTask):
    name = "mmlu_adv"
    labels_list = ["A", "B", "C", "D"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "adversarial",
        "validation": "adversarial",
        "test": "test",
    }
    label_column_name = "answer"
    id2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    def load_dataset(self, split, subset=None) -> Dataset:
        return datasets.load_dataset("rbelanec/mmlu_paraphrases", subset, split="adversarial")

    def preprocessor(self, example):
        input_texts = [
            "Select the best answer from the given options. Respond only with the letter corresponding to the correct choice.",
            example["input_text"],
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        if self.split == "test":
            return self.apply_test_template(
                {
                    "content": "\n".join(input_texts) + "\n",
                    "target": " ".join(label_texts),
                    "role": "user",
                }
            )
        else:
            return self.apply_template(
                {
                    "content": "\n".join(input_texts) + "\n",
                    "target": " ".join(label_texts),
                    "role": "user",
                }
            )
