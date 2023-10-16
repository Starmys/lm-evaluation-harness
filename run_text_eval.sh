#!/bin/bash

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=8,ignore_layers=0,ignore_components=ffn.up_proj \
    --tasks piqa,boolq \
    --device cuda:0 \
    --batch_size 8 \
    --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a8-ig0-up.json

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=8,ignore_layers=0,ignore_components=ffn.down_proj \
    --tasks piqa,boolq \
    --device cuda:0 \
    --batch_size 8 \
    --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a8-ig0-down.json

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=8,ignore_layers=0,ignore_components="ffn.up_proj ffn.down_proj" \
    --tasks piqa,boolq \
    --device cuda:0 \
    --batch_size 8 \
    --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a8-ig0-ffn.json

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=32,ignore_layers=0,ignore_components=ffn.up_proj \
    --tasks piqa,boolq \
    --device cuda:0 \
    --batch_size 8 \
    --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a16-ig0-up.json

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=32,ignore_layers=0,ignore_components=ffn.down_proj \
    --tasks piqa,boolq \
    --device cuda:0 \
    --batch_size 8 \
    --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a16-ig0-down.json

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=32,ignore_layers=0,ignore_components="ffn.up_proj ffn.down_proj" \
    --tasks piqa,boolq \
    --device cuda:0 \
    --batch_size 8 \
    --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a16-ig0-ffn.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w16a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=4 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a4.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=8 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a8.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=16 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-8bit.pt",w_bits=8,a_bits=4 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w8a4.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-8bit.pt",w_bits=8,a_bits=8 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w8a8.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-8bit.pt",w_bits=8,a_bits=16 \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w8a16.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b",trust_remote_code=True \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=4,smooth_checkpoint=fake \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a4-sf.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=8,smooth_checkpoint=fake \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a8-sf.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-4bit.pt",w_bits=4,a_bits=16,smooth_checkpoint=fake \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w4a16-sf.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-8bit.pt",w_bits=8,a_bits=4,smooth_checkpoint=fake \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w8a4-sf.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-8bit.pt",w_bits=8,a_bits=8,smooth_checkpoint=fake \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w8a8-sf.json

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="/home/chengzhang/models/mpt/mpt-7b,trust_remote_code=True" \
#     --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/mpt7b-8bit.pt",w_bits=8,a_bits=16,smooth_checkpoint=fake \
#     --tasks piqa,boolq \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path /home/chengzhang/Multimodal-Quantization/evaluation/Harness/mpt-7b-gptq-w8a16-sf.json
