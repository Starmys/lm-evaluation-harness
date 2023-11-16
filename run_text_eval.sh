#!/bin/bash

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llama-hf/llama-7b_hf" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llama7b-4bit.pt",w_bits=4,a_bits=4,act_quant_func="lut",act_quant_dim=1,act_token_split=1,outliers_thres=0 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llama-7b-gptq-w4a4-cbrt-outliers-perchannel.json

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b" \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-c4-4bit.pt",w_bits=4,a_bits=4,act_quant_func="lut.text",act_quant_dim=1,act_token_split=0,outliers_thres=0 \
    --tasks piqa,boolq \
    --device cuda:0 \
    --batch_size 8 \
    --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-c4-w4a4-lut.json
