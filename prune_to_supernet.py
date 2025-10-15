# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

import os
import argparse
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from src import dist_utils
from src.data_utils import get_data
from src.model_utils import get_hidden_size, get_head_size, get_layers, get_down_proj_params, get_o_proj_params
from src.pruner import Pruner
from src.common_utils import none_or_str


def parse_args():
    parser = argparse.ArgumentParser(description="Tyr-the-Pruner: Prune-to-Supernet.")
    # Model params
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The name or path to the model being pruned")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="The name or path to the tokenizer. By default use model tokenizer.")
    parser.add_argument("--prunable_modules", type=str, required=True, help="Regex for modules to prune")
    parser.add_argument("--pre_block_modules", nargs="+", type=str, required=True, help="Names of modules before transformer blocks")
    parser.add_argument("--block_modules", type=str, required=True, help="Name of transformer modules")
    # Data params
    parser.add_argument("--calibration_data", type=str, required=True, help="The name or dataset or path used for calibration.")
    parser.add_argument("--calibration_tokens", default=int(2**23), type=int, help="Number of tokens for calibration.")
    parser.add_argument("--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences.")
    # Sparsification params
    parser.add_argument("--sparsity", required=True, type=float)
    parser.add_argument("--supernet_dir", default=None, type=none_or_str)
    parser.add_argument("--supernet_config", default=None, type=none_or_str)
    parser.add_argument("--weights_diff_mha", default=None, type=int)
    parser.add_argument("--weights_diff_mlp", default=None, type=int)
    parser.add_argument("--error_accumulation", action="store_true", help="whether to accumulate errors")
    parser.add_argument("--num_sparsity_levels", default=9, type=int)
    # Save params
    parser.add_argument("--save_dir", type=str, required=True, help="where to save sparse supernet.")
    # Misc params
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--seed", default=0, type=int, help="random seed.")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="whether to load model with the use of `low_cpu_mem_usage`")
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--verbose", action="store_true", help="whether to log progress.")
    args = parser.parse_args()
    return args


def main():
    import datetime
    args = parse_args()
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=2400))
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # init device
    device = f"cuda:{rank}"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )
    if not args.cpu_offload_modules:
        model = model.to(device)
    
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=False)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=False)
    
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or min(model.config.max_position_embeddings, 4096)
    calibration_data = get_data(args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True)
    
    # take slice (if running on multiple workers)
    if dist_utils.is_dist_available_and_initialized():
        num_seq_per_rank = len(calibration_data) // world_size
        calibration_data = calibration_data[rank * num_seq_per_rank : (rank + 1) * num_seq_per_rank]
    calibration_data = [([], {"input_ids": input_ids}) for input_ids in calibration_data]
    dist.barrier()
    
    # Pruner
    pruner = Pruner(
        model,
        calibration_data,
        prunable_modules=args.prunable_modules,
        pre_block_modules=args.pre_block_modules,
        block_modules=args.block_modules,
        save_dir=args.save_dir,
        error_accumulation=args.error_accumulation,
        device=device,
        cpu_offload_modules=args.cpu_offload_modules,
        cpu_offload_activations=args.cpu_offload_activations,
        verbose=args.verbose,
    )
    

    hidden_size = get_hidden_size(model)
    head_size = get_head_size(model)
    args.weights_diff_mlp = max(args.weights_diff_mlp, 32 * hidden_size)
    args.weights_diff_mha = max(args.weights_diff_mha, int(head_size * hidden_size))

    assert (args.supernet_dir is None) == (args.supernet_config is None), "supernet_dir and supernet_config should be None or both not None."

    if args.supernet_dir is not None:
        o_proj_params = get_o_proj_params(model)
        down_proj_params = get_down_proj_params(model)
        layer_num = len(get_layers(model))
        o_proj_sparsity_level = [0] * layer_num
        down_proj_sparsity_level = [0] * layer_num
        with open(os.path.join(args.supernet_config), "r") as f:
            for line in f:
                layer_name, level = line.split(":")
                layer_id = int(layer_name.strip().split(".")[2])
                if 'down_proj' in layer_name:
                    down_proj_sparsity_level[layer_id] = int(level.strip())
                else:
                    o_proj_sparsity_level[layer_id] = int(level.strip())
        meta_configs = torch.load(os.path.join(args.supernet_dir, "metadata.pth"))
        weights_diff_mlp = meta_configs['weights_diff_mlp']
        weights_diff_mha = meta_configs['weights_diff_mha']
        init_sparsity_mha = meta_configs['mha_sparsity']
        init_sparsity_mlp = meta_configs['ffn_sparsity']
        args.mha_sparsity = [(init_sparsity_mha[layer_i] * o_proj_params + level_i * weights_diff_mha) / o_proj_params for layer_i, level_i in enumerate(o_proj_sparsity_level)]
        args.ffn_sparsity = [(init_sparsity_mlp[layer_i] * down_proj_params + level_i * weights_diff_mlp) / down_proj_params for layer_i, level_i in enumerate(down_proj_sparsity_level)]
    else:
        args.mha_sparsity = [args.sparsity] * len(get_layers(model))
        args.ffn_sparsity = [args.sparsity] * len(get_layers(model))

    print("args.mha_sparsity: ", args.mha_sparsity)
    print("args.ffn_sparsity: ", args.ffn_sparsity)
    
    # Prepare save dir
    if dist_utils.is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            {"sparsity": args.sparsity, 
             "mha_sparsity": args.mha_sparsity,
             "ffn_sparsity": args.ffn_sparsity,
             "weights_diff_mlp": args.weights_diff_mlp, 
             "weights_diff_mha": args.weights_diff_mha, 
             "num_sparsity_levels": args.num_sparsity_levels},
            os.path.join(args.save_dir, "metadata.pth"),
        )
    
    dist.barrier()
    t1 = time.perf_counter()
    pruner.prune(args.mha_sparsity, args.ffn_sparsity, args.weights_diff_mha, args.weights_diff_mlp, args.num_sparsity_levels)
    t2 = time.perf_counter()
    dist_utils.print_on_main(f"Pruning to supernet took {(t2 - t1)} s.")


if __name__ == "__main__":
    main()
