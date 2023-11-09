#!/bin/bash

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llama-hf/llama-7b_hf" \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llama-7b-gptq-w16a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llama-hf/llama-7b_hf" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llama7b-4bit.pt",w_bits=4,a_bits=4 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llama-7b-gptq-w4a4-rtnc.json

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="/home/chengzhang/models/llama-hf/llama-7b_hf" \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llama7b-4bit.pt",w_bits=4,a_bits=-4 \
    --tasks piqa,boolq \
    --device cuda:0 \
    --batch_size 8 \
    --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llama-7b-gptq-w4a4-nuoc.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-c4-4bit.pt",w_bits=4,a_bits=-4 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-c4-w4a4-nuc.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-c4-4bit.pt",w_bits=4,a_bits=4 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-c4-w4a4-rtnc.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b",trust_remote_code=True \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-textv-4bit.pt",w_bits=4,a_bits=32 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-textv-w4a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b",trust_remote_code=True \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-textvqa-4bit.pt",w_bits=4,a_bits=32 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-textvqa-w4a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b",trust_remote_code=True \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-c4-4bit.pt",w_bits=4,a_bits=32 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-c4-w4a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b",trust_remote_code=True \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-textv-8bit.pt",w_bits=8,a_bits=32 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-textv-w8a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b",trust_remote_code=True \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-textvqa-8bit.pt",w_bits=8,a_bits=32 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-textvqa-w8a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/llava/llava-v1.5-7b",trust_remote_code=True \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-c4-8bit.pt",w_bits=8,a_bits=32 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/llava-v1.5-7b-gptq-c4-w8a16.json
