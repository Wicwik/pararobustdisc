import functools
import datasets
import torch

from typing import Mapping

from typing import List, Dict, Callable
from datasets import Dataset

from transformers import EvalPrediction, PreTrainedTokenizer


class AbstractTask:
    name: str = NotImplemented
    preprocessor = NotImplemented
    metrics: List[Callable] = NotImplemented
    metric_names: List[str] = NotImplemented
    seed: int = NotImplemented
    labels_list: List[str] = None
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    small_datasets_without_all_splits: List[str] = []
    large_data_without_all_splits: List[str] = ["mmlu", "mmlu_paraphrases"]
    id2label: Dict[int, str] = NotImplemented
    label_column_name: str = NotImplemented

    def __init__(self, tokenizer: PreTrainedTokenizer, seed: int = 42) -> None:
        self.seed = seed
        self.tokenizer = tokenizer

    def postprocessor(self, preds, labels):
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True, return_tensors="pt"
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, return_tensors="pt"
        )

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        return decoded_preds, decoded_labels

    # get maximum token lenght from labels
    def get_max_target_length(self, default_max_length) -> int:
        if self.labels_list is not None:
            return max(
                [len(self.tokenizer.encode(label)) for label in self.labels_list]
            )
        return default_max_length

    def check_n_obs(self, n_obs, total_size) -> int:
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
        return n_obs

    # generates indices of the dataset randomly with seed (if same seed and data provided we will still get the same shuffle, no matter how many times initialized)
    def shuffled_indices(self, dataset) -> List[int]:
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset: Dataset, n_obs=None):
        num_samples = len(dataset)

        if n_obs >= num_samples:
            return dataset

        return dataset.train_test_split(
            train_size=n_obs / num_samples,
            seed=self.seed,
            stratify_by_column=self.label_column_name,
        )["train"]

    def get_splits(self, split, dataset: Dataset, validation_size):
        if split == "validation":
            return dataset.train_test_split(
                train_size=validation_size,
                test_size=1 - validation_size,
                seed=self.seed,
                stratify_by_column=self.label_column_name,
                shuffle=True,
            )["train"]

        return dataset.train_test_split(
            train_size=validation_size,
            test_size=1 - validation_size,
            seed=self.seed,
            stratify_by_column=self.label_column_name,
            shuffle=True,
        )["test"]

    def map_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            functools.partial(self.preprocessor),
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc=f"Running {self.name}_preprocessor on dataset",
        )

    def load_dataset(self, split: int) -> Dataset:
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def apply_test_template(self, examples):
        return {
            "text": self.tokenizer.apply_chat_template(
                [examples], tokenize=False, add_generation_prompt=True
            ),
            "target": examples["target"]
        }

    def apply_template(self, examples):
        return {
            "text": self.tokenizer.apply_chat_template(
                [examples, {"role": "assistant", "content": examples["target"]}],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    def get_compute_metrics(
        self,
        postprocess=True,
    ) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics(eval_preds: EvalPrediction) -> Dict:
            preds, labels = eval_preds

            if postprocess:
                preds[preds == -100] = self.tokenizer.pad_token_id
                labels[labels == -100] = self.tokenizer.pad_token_id

                decoded_preds, decoded_labels = self.postprocessor(preds, labels)
            else:
                decoded_preds = preds
                decoded_labels = labels

            # print("compute_metrics:", decoded_preds, decoded_labels)

            metrics = {}
            for m, n in zip(self.metrics, self.metric_names):
                if "f1" in n:
                    metrics.update(m(decoded_preds, decoded_labels, self.labels_list))
                else:
                    metrics.update(m(decoded_preds, decoded_labels))

            return metrics

        return compute_metrics

    def get(
        self,
        split,
        n_obs=None,
        split_validation_test=False,
    ) -> Dataset:
        self.split = split

        if (
            split_validation_test
            and self.name in self.small_datasets_without_all_splits
            and split != "train"
        ):
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            dataset = self.get_splits(split, dataset, 0.5)

            if n_obs:
                dataset = self.subsample(dataset, n_obs)

        elif (
            split_validation_test
            and self.name in self.large_data_without_all_splits
            and split != "test"
        ):
            mapped_split = self.split_to_data_split["train"]
            dataset = self.load_dataset(split=mapped_split)
            dataset = self.get_splits(split, dataset, 1000 / len(dataset))

            if n_obs:
                dataset = self.subsample(dataset, n_obs)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split).shuffle(seed=self.seed)

            if n_obs:
                dataset = self.subsample(dataset, n_obs)

        return self.map_dataset(dataset)
