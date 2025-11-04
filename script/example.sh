# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

#!/bin/bash

export HIP_VISIBLE_DEVICES=0
export SSL_CERT_DIR='/etc/ssl/certs'
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
export HF_DATASETS_CACHE=/path/to/ample/storage/space
export TRANSFORMERS_CACHE=/path/to/ample/storage/space
export HF_ALLOW_CODE=1
export NCCL_TIMEOUT=3600

CALIBRATION_DATA=fineweb_edu
SEARCH_DATA=fineweb_edu

GENERATIONS=(50 50 50 50)
OFFSPRINGS=(128 128 128 128)

for MODEL in Llama-2-7b-hf; do 
    for SPARSITY in 0.5 0.375 0.25 0.125; do
        WEIGHTS_DIFF_MLP=5636096
        WEIGHTS_DIFF_MHA=2097152
        for R in 1 2 3 4; do

            if [ "$R" -eq 1 ]; then
                SUPERNET_DIR=None
                SUPERNET_CONFIG=None
            else
                SUPERNET_DIR="./Cached-Supernets/${MODEL}-${SPARSITY}-iteration$((R-1))"
                SUPERNET_CONFIG="./Cached-Supernets/${MODEL}-${SPARSITY}-iteration$((R-1))/${MODEL}-${SPARSITY}.txt"
            fi

            # algorithm 1: prune to supernet
            torchrun --nnodes=1 --nproc_per_node=1 --master_port=29509 prune_to_supernet.py \
                --model_name_or_path meta-llama/${MODEL} \
                --prunable_modules '^(?!.*(?:embedding|emb|head)).+$' \
                --pre_block_modules model.embed_tokens model.rotary_emb \
                --block_modules model.layers \
                --calibration_data ${CALIBRATION_DATA} \
                --calibration_sequence_length 4096 \
                --calibration_tokens 4194304 \
                --dtype bfloat16 \
                --low_cpu_mem_usage \
                --attn_implementation sdpa \
                --cpu_offload_modules \
                --cpu_offload_activations \
                --verbose \
                --sparsity ${SPARSITY} \
                --error_accumulation \
                --supernet_dir ${SUPERNET_DIR} \
                --supernet_config ${SUPERNET_CONFIG} \
                --weights_diff_mlp ${WEIGHTS_DIFF_MLP} \
                --weights_diff_mha ${WEIGHTS_DIFF_MHA} \
                --save_dir ./Cached-Supernets/${MODEL}-${SPARSITY}-iteration${R} \
                --num_sparsity_levels 9
            
            # algorithm 2: search sparsity distribution
            python3 search_sparsity_dist.py \
                --model_name_or_path meta-llama/${MODEL} \
                --calibration_data ${SEARCH_DATA} \
                --calibration_sequence_length 4096 \
                --calibration_tokens 4194304 \
                --eval_datasets wikitext2 \
                --eval_every 5 \
                --eval_tokens 524288 \
                --eval_sequence_length 4096 \
                --fitness_fn sparse_kl \
                --kl_topk 8192 \
                --dtype bfloat16 \
                --attn_implementation sdpa \
                --offspring 128 \
                --generations ${GENERATIONS[R-1]} \
                --tokens_per_selection 2048 16384 131072 \
                --survivors_per_selection 16 4 2 \
                --sparse_weights_path ./Cached-Supernets/${MODEL}-${SPARSITY}-iteration${R} \
                --configuration_name ${MODEL}-${SPARSITY}.txt
            
            WEIGHTS_DIFF_MHA=$((WEIGHTS_DIFF_MHA / 2))
            WEIGHTS_DIFF_MLP=$((WEIGHTS_DIFF_MLP / 2))
        
        done
    done
done

