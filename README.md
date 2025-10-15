# Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization

Paper: https://arxiv.org/abs/2503.09657

## Setup

```text
accelerate                    1.4.0
datasets                      2.21.0
hf_transfer                   0.1.9
hf-xet                        1.1.7
huggingface-hub               0.34.4
numpy                         1.26.4
pytorch-triton-rocm           3.1.0
tabulate                      0.9.0
torch                         2.5.1+rocm6.2
tokenizers                    0.19.1
tqdm                          4.67.1
tqdm-multiprocess             0.0.11
transformers                  4.44.0
triton                        3.2.0
```

## Example

```bash
bash script/example.sh
```

```bash
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
```

## Acknowledgements

This project makes use of open-source code and ideas from the following repositories:

- [EvoPress](https://github.com/IST-DASLab/EvoPress): Evolutionary structured pruning and compression for transformer models.
- [FLAP](https://github.com/CASIA-IVA-Lab/FLAP): Fast Layer-wise Adaptive Pruning framework for efficient deep networks.
- [OSSCAR](https://github.com/mazumder-lab/OSSCAR): Open-Source framework for Sparse Compression and Acceleration of deep neural networks.

We sincerely thank the authors and contributors of these excellent works for making their code publicly available.

## Cite

```
@inproceedings{Li2025TyrThePruner,
  title        = {Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization},
  author       = {Li, Guanchen and Xu, Yixing and Li, Zeping and Liu, Ji and Yin, Xuanwu and Li, Dong and Barsoum, Emad},
  booktitle    = {NeurIPS 2025},
  year         = {2025},
  url          = {https://neurips.cc/virtual/2025/poster/115807}
}
```
