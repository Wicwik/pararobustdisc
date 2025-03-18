from peft import AutoPeftModelForCausalLM

saves = ["pt_03172025202530_mmlu_paraphrases_gemma-2-9b-it_best", "pt_03172025202530_mmlu_gemma-2-9b-it_best", "pt_03172025184408_mmlu_paraphrases_meta-llama-3.1-8b-instruct_best", "pt_03172025184408_mmlu_meta-llama-3.1-8b-instruct_best"]

for save in saves:
    model = AutoPeftModelForCausalLM.from_pretrained(f"saves/{save}")

    print(model)

    model.push_to_hub(f"rbelanec/{save}")