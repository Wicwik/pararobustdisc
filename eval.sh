#!/bin/bash

set -euo pipefail

run_llama=false
run_gemma=false
run_lora=false
run_pt=false

llama(){
    if $run_pt
    then
        local models_to_eval=(rbelanec/pt_03172025184408_mmlu_meta-llama-3.1-8b-instruct rbelanec/pt_03172025184408_mmlu_paraphrases_meta-llama-3.1-8b-instruct)

        for i in "${models_to_eval[@]}";
        do
            echo $i;
            python eval.py configs/pt/llama/mmlu.yaml $i;
        done
    fi

    if $run_lora
    then
        local models_to_eval=(rbelanec/lora_03182025094215_mmlu_meta-llama-3.1-8b-instruct rbelanec/lora_03182025094215_mmlu_paraphrases_meta-llama-3.1-8b-instruct)

        for i in "${models_to_eval[@]}";
        do
            echo $i;
            python eval.py configs/lora/llama/mmlu.yaml $i;
        done
    fi
}

gemma(){
    if $run_pt
    then
        local models_to_eval=(rbelanec/pt_03172025202530_mmlu_gemma-2-9b-it rbelanec/pt_03172025202530_mmlu_paraphrases_gemma-2-9b-it)

        for i in "${models_to_eval[@]}";
        do
            echo $i;
            python eval.py configs/pt/gemma/mmlu.yaml $i;
        done
    fi

    if $run_lora
    then
        local models_to_eval=(rbelanec/lora_03182025100228_mmlu_gemma-2-9b-it rbelanec/lora_03182025100228_mmlu_paraphrases_gemma-2-9b-it)

        for i in "${models_to_eval[@]}";
        do
            echo $i;
            python eval.py configs/lora/gemma/mmlu.yaml $i;
        done
    fi
}

usage(){
>&2 cat << EOF
Usage: $0
   [ -l | --llama ]
   [ -g | --gemma ]
   [ -o | --lora  ]
   [ -p | --pt    ]
EOF
exit 1
}

args=$(getopt -o lghop --long llama,gemma,lora,pt,help -- "$@")

if [[ $? -gt 0 ]]; then
  usage
fi

eval set -- ${args}

while :
do
  case $1 in
    -l | --llama)   run_llama=true     ; shift   ;;
    -g | --gemma)   run_gemma=true     ; shift   ;;
    -o | --lora)    run_lora=true      ; shift   ;;
    -p | --pt)      run_pt=true        ; shift   ;;
    -h | --help)    usage          ; shift   ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done

if $run_llama
then
    llama
fi
 
if $run_gemma
then
    gemma
fi