from datasets import load_dataset

dataset = load_dataset(
    "csv", data_files={"train_gemma_dpo": "train_gemma_dpo_new.csv"}
).class_encode_column("answer")
print(dataset.push_to_hub("rbelanec/mmlu_paraphrases"))
