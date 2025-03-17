from datasets import load_dataset

dataset = load_dataset("csv", data_files={"train_gemma_dpo": "train_gemma_dpo.csv"})
print(dataset.push_to_hub("rbelanec/mmlu_paraphrases"))