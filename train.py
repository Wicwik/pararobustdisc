import argparse, os, torch, wandb

from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    DataCollatorWithPadding,
)
from peft import get_peft_model, PromptTuningConfig, LoraConfig
from trl import SFTConfig, ModelConfig, SFTTrainer

from args import DataConfig
from tasks import AutoTask

import numpy as np

from dataclasses import dataclass, field


# workaround for HF Parser https://github.com/huggingface/transformers/issues/34834
@dataclass
class CustomLoraConfig(LoraConfig):
    init_lora_weights: bool = field(default=True)
    layers_to_transform: int = field(default=None)
    loftq_config: dict = field(default_factory=dict)


def get_run_name(timestamp, config_filename, dataset_name, model_name_or_path):
    method = config_filename.split("/")[1]
    model = model_name_or_path.split("/")[-1].lower()
    return f"{method}_{timestamp}_{dataset_name}_{model}"


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

argparse_parser = argparse.ArgumentParser(
    prog="Train discriminator model on paraphrases",
    description="Train model on paraphrases from chosen dataset.",
)

argparse_parser.add_argument(
    "filename", help="Filename of a config in yaml format to run."
)
argparse_parser.add_argument(
    "--print_data",
    action="store_true",
    help="Prints data structure of a first sample from train and valid sets.",
)
args = argparse_parser.parse_args()

print(args.filename)

peft_config_type = PromptTuningConfig if "pt" in args.filename else CustomLoraConfig

hfparser = HfArgumentParser((SFTConfig, peft_config_type, DataConfig))

sft_config, peft_config, data_config = hfparser.parse_yaml_file(
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

for dataset_name in data_config.dataset_names:
    sft_config.run_name = get_run_name(
        timestamp, args.filename, dataset_name, model_config.model_name_or_path
    )
    sft_config.output_dir = f"saves/{sft_config.run_name}"

    set_seed(sft_config.seed)
    np.random.seed(seed=sft_config.seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if "llama" in model_config.model_name_or_path.lower():
        model.active_adapters = [
            "default"
        ]  # fix because llama has some active adapters for some reason

        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})

        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model = get_peft_model(model, peft_config=peft_config)
    print(peft_config)

    if "pt" in args.filename:
        indices = np.random.permutation(range(5000))[:100]

        word_embedding_weights = (
            model.word_embeddings(torch.LongTensor(indices).to("cuda")).detach().clone()
        )

        model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            word_embedding_weights
        )

        print(
            "current PT weights:",
            model.prompt_encoder.default.embedding.weight,
            model.prompt_encoder.default.embedding.weight.shape,
        )

    model.print_trainable_parameters()

    print(f"task: {dataset_name}")

    train_dataset = AutoTask.get(dataset_name, tokenizer, seed=sft_config.seed).get(
        split="train",
        n_obs=data_config.max_train_samples,
        split_validation_test=data_config.split_validation_test,
    )
    valid_dataset = AutoTask.get(dataset_name, tokenizer, seed=sft_config.seed).get(
        split="validation",
        n_obs=data_config.max_valid_samples,
        split_validation_test=data_config.split_validation_test,
    )

    if args.print_data:
        print("Train data")
        print(train_dataset)
        print(train_dataset["text"][0])
        print(tokenizer(train_dataset["text"][0]))

        print("Valid data")
        print(valid_dataset)
        print(valid_dataset["text"][0])
        print(tokenizer(valid_dataset["text"][0]))

        continue

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    save_name = f"{sft_config.output_dir}_best"

    model.save_pretrained(save_name)

    if wandb.run is not None:
        artifact = wandb.Artifact(name=sft_config.run_name, type="weights")
        artifact.add_dir(local_path=save_name)
        wandb.run.log_artifact(artifact)
        wandb.log(data={})

        wandb.finish()
