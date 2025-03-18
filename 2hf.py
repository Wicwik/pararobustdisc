from peft import AutoPeftModelForCausalLM
import argparse

argparse_parser = argparse.ArgumentParser(
    prog="Upload model to HF HUB.",
    description="Tool for uploading single model from saves to hub.",
)

argparse_parser.add_argument(
    "save", help="Directory to a saved HF model."
)

args = argparse_parser.parse_args()

model = AutoPeftModelForCausalLM.from_pretrained(f"{args.save}").to("cuda")

print(model)

hub_name = args.save.split("/")[-1]
hub_name = hub_name.replace("_best", "")

model.push_to_hub(f"rbelanec/{hub_name}")