# The name of experiment
name=VQAv2_Our

output=snap/$name

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PYTHONPATH=$PYTHONPATH:./src \
torchrun --nproc_per_node=$1 \
    --master_port 29500 \
    src/vqacl.py \
        --distributed --multiGPU \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 3e-5 \
        --epochs 3 \
        --num_workers 4 \
        --backbone '/root/autodl-tmp/VQACL/VL-T5/models/t5-base'\
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 24 \
        --valid_batch_size 32 \
        --from_scratch \
        --memory \
        --m_size 5000 \
        --comp_cate G-1  \
        --fp16 
