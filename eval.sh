#!/bin/bash

models_to_eval=(rbelanec/pt_03172025184408_mmlu_meta-llama-3.1-8b-instruct rbelanec/pt_03172025184408_mmlu_paraphrases_meta-llama-3.1-8b-instruct)

for i in "${models_to_eval[@]}";
do
    echo $i;
    python eval.py configs/pt/llama/mmlu.yaml $i;
done