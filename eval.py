import argparse, os, torch

from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    EvalPrediction,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    pipeline,
)
from peft import (
    AutoPeftModelForCausalLM,
)
from trl import SFTConfig, ModelConfig

from args import DataConfig
from tasks import AutoTask

from tqdm import tqdm

import numpy as np

import pandas as pd


def get_run_name(timestamp, config_filename, dataset_name, model_name_or_path):
    method = config_filename.split("/")[1]
    model = model_name_or_path.split("/")[-1].lower()
    return f"{method}_{timestamp}_{dataset_name}_{model}"


def predict(test_dataset, model, tokenizer, labels_list):
    y_pred = []
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=16,
        do_sample=False,
        top_p=None,
        temperature=None,
        use_cache=False,
        device="cuda",
    )

    for x_test in tqdm(test_dataset["text"]):

        # print(x_test)
        result = pipe(x_test)
        # print(result)

        if "llama" in model.config.name_or_path.lower():
            answer = (
                result[0]["generated_text"]
                .split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]
                .strip()
            )
        elif "gemma" in model.config.name_or_path.lower():
            answer = (
                result[0]["generated_text"]
                .split("<start_of_turn>model")[-1]
                .split("\n")[1]
                .strip()
            )
        
        for label in labels_list:
            if label.lower() == answer.lower():
                y_pred.append(label)
                break
        else:
            y_pred.append("none")
            print("not matching labels:", result, answer)

        # print(answer)

    return y_pred

def evaluate(y_pred, y_true, compute_metrics, prefix="eval"):
    metrics = compute_metrics(EvalPrediction(y_pred, y_true))

    return {f"{prefix}/{k}": v for k, v in metrics.items()}


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

argparse_parser = argparse.ArgumentParser(
    prog="Train discriminator model on paraphrases",
    description="Train model on paraphrases from chosen dataset.",
)

argparse_parser.add_argument(
    "filename", help="Filename of a config in yaml format to run."
)
argparse_parser.add_argument(
    "model_or_path", help="HF Hub model or path to load and eval."
)
argparse_parser.add_argument(
    "--print_data",
    action="store_true",
    help="Prints data structure of a first sample from train and valid sets.",
)
argparse_parser.add_argument(
    "--no_peft",
    action="store_true",
    help="Do not use PEFT.",
)
args = argparse_parser.parse_args()

print(args.filename)

hfparser = HfArgumentParser((SFTConfig, DataConfig))

sft_config, data_config = hfparser.parse_yaml_file(
    str(args.filename), allow_extra_keys=True
)

hfparser = HfArgumentParser((ModelConfig))
model_config = hfparser.parse_yaml_file(str(args.filename), allow_extra_keys=True)[0]

torch_dtype = (
    model_config.torch_dtype
    if model_config.torch_dtype in ["auto", None]
    else getattr(torch, model_config.torch_dtype)
)

os.environ["WANDB_ENTITY"] = "rbelanec"
os.environ["WANDB_PROJECT"] = "robustifying-models-for-benchmarks"

full_test_results = {args.model_or_path.split("/")[-1]: {}}

for dataset_name in data_config.dataset_names:
    sft_config.run_name = get_run_name(
        timestamp, args.filename, dataset_name, model_config.model_name_or_path
    )
    sft_config.output_dir = f"saves/{sft_config.run_name}"

    set_seed(sft_config.seed)
    np.random.seed(seed=sft_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if args.no_peft:
        model = AutoModelForCausalLM.from_pretrained(args.model_or_path).to("cuda")

        model.active_adapters = [
            "default"
        ]  # fix because llama has some active adapters for some reason
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_or_path).to("cuda")

        if "pt" in args.filename:
            print(
                "current PT weights:",
                model.prompt_encoder.default.embedding.weight,
                model.prompt_encoder.default.embedding.weight.shape,
            )
        
        model.print_trainable_parameters()

    if "llama" in model_config.model_name_or_path.lower():
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})

        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"task: {dataset_name}")

    test_dataset = AutoTask.get(dataset_name, tokenizer, seed=sft_config.seed).get(
        split="test",
        n_obs=data_config.max_test_samples,
        split_validation_test=data_config.split_validation_test,
    )

    if args.print_data:
        print("Test data")
        print(test_dataset)
        print(test_dataset["text"][0])
        print(test_dataset["target"][0])

        exit()

    compute_metrics = AutoTask.get(dataset_name, tokenizer, seed=sft_config.seed).get_compute_metrics(postprocess=False)



    test_results = evaluate(
        predict(
            test_dataset,
            model,
            tokenizer,
            AutoTask.get(dataset_name, tokenizer, seed=sft_config.seed).labels_list,
        ),
        test_dataset["target"],
        compute_metrics,
        prefix="test",
    )

    print(test_results)

    full_test_results[args.model_or_path.split("/")[-1]][dataset_name] = test_results

    del model, tokenizer

df = pd.DataFrame.from_dict(full_test_results)

df.to_csv(f"{timestamp}_{args.model_or_path.split('/')[-1]}_test_results.csv")
