import argparse

from transformers import HfArgumentParser
from peft import PromptTuningConfig, LoraConfig

from trl import SFTConfig, ModelConfig

from args import DataConfig

argparse_parser = argparse.ArgumentParser(
    prog="Train discriminator model on paraphrases",
    description="Train model on paraphrases from chosen dataset.",
)

argparse_parser.add_argument("filename", help="Filename of a config in yaml format to run.")
args = argparse_parser.parse_args()

print(args.filename)

hfparser = HfArgumentParser(
    (SFTConfig, ModelConfig, PromptTuningConfig, DataConfig)
)

sft_config, model_config, peft_config, data_config = hfparser.parse_yaml_file(str(args.filename))

# print(sft_config, model_config, peft_config, data_config)