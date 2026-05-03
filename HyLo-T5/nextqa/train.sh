#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 

port=29508
m_size=500
epoch=10
seed=6666
output=snap/nextqa/checkpoint

torchrun --nproc_per_node=1 --master_port $port \
    nextqa/nextqa_CL.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --lr 3e-4 \
    --clip_grad_norm 5 \
    --num_workers 4 \
    --backbone '/HyLoVQA-main/HyLoVQA-T5/models/t5-base' \
    --num_beams 5 \
    --valid_batch_size 100 \
    --epochs $epoch \
    --batch_size 80 \
    --from_scratch \
    --memory \
    --m_size 500 \
    --comp_cate G-1 \
    --ifseed \
    --seed $seed \
    --proto_beta 0.5 \
    --proto_alpha 0.3 \
    --output $output
