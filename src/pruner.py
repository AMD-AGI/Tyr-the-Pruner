# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

import os
from tqdm import tqdm, trange
from typing import Iterable, Dict, List, Any, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from src import dist_utils
from src.local_pruner import LocalPruner
from src.common_utils import to, maybe_first_element
from src.model_utils import InputCollector, ForwardInterrupt, LINEAR_LAYERS, select_layers


class Pruner:

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        prunable_modules: str,
        pre_block_modules: List[str],
        block_modules: str,
        save_dir: Union[str, os.PathLike],
        error_accumulation: bool = False,
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.prunable_modules = prunable_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.error_accumulation = error_accumulation
        self.save_dir = save_dir
        self.device = device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations
        self.verbose = verbose

    @torch.no_grad()
    def prune(self, mha_sparsity, ffn_sparsity, weights_diff_mha: int, weights_diff_mlp: int, num_sparsity_levels: int):
        """
        Args:
            sparsity: target average sparsity
            weights_diff: difference in number of non-zero weights between sparsity levels
            num_sparsity_levels: number of substructures in single supernet layer
        """
        device = self.device or next(self.model.parameters()).device
        # prepare pre blocks modules
        blocks = self._get_submodule(self.block_modules)
        pre_blocks = [self._get_submodule(module_name) for module_name in self.pre_block_modules]
        blocks[0] = blocks[0].to(device)
        for module in pre_blocks:
            module.to(device)
        # Cache
        if hasattr(self.model.config, "use_cache"):
            use_cache = self.model.config.use_cache
            self.model.config.use_cache = False
        # Input preparation #
        blocks[0] = InputCollector(blocks[0], cpu_offload=self.cpu_offload_activations)
        for inp_args, inp_kwargs in self.data_loader:
            try:
                self.model(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            except ForwardInterrupt:
                pass
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()

        # offload pre_blocks
        if self.cpu_offload_modules:
            for module in pre_blocks:
                module.cpu()

        # Block pruning #
        for block_id, block in enumerate(blocks):

            # TODO change to logging
            if self.verbose:
                dist_utils.print_on_main(f"Processing {self.block_modules} {block_id}/{len(blocks)}.")
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f"{self.block_modules}.{block_id}."
            layers = select_layers(self.model, layer_prefix, self.prunable_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(layers)

            for inp_args, inp_kwargs in tqdm(zip(input_args, input_kwargs), desc="Generate layer input activation..."):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

            for _, h in hooks.items():
                h.remove()

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()

            self._prune_group(handles, block_id, mha_sparsity, ffn_sparsity, weights_diff_mha, weights_diff_mlp, num_sparsity_levels)

            for inp_args, inp_kwargs in tqdm(zip(input_args, input_kwargs), desc="Generate layer output activation..."):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                out = maybe_first_element(out)
                if self.cpu_offload_activations:
                    out = out.cpu()
                if len(inp_args) > 0:
                    inp_args[0].data = out
                elif "hidden_states" in inp_kwargs:
                    inp_kwargs["hidden_states"] = out
                else:
                    raise ValueError("Unsupported block input format.")

            if self.cpu_offload_modules:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = use_cache

    def _get_submodule(self, module_name: str):
        return self.model.get_submodule(module_name)

    def _prepare_hooks_and_handles(self, layers: Dict[str, nn.Module]):
        handles = {}
        hooks = {}
        for layer_name, layer in layers.items():
            if any(proj in layer_name for proj in ("self_attn.o_proj", "self_attn.out_proj", "mlp.down_proj")):
                
                def update_handle_hook(name):
                    def _hook(_, inp, out):
                        handles[name].update(inp[0])
                    return _hook

                handles[layer_name] = self._create_handle(layer, layer_name)
                hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))
        return handles, hooks

    def _create_handle(self, layer, layer_name):
        return LocalPruner(layer, layername=layer_name, num_heads=self.model.config.num_attention_heads)

    def _prune_group(self, handles: Dict[str, LocalPruner], block_id, mha_sparsity, ffn_sparsity, weights_diff_mha: int, weights_diff_mlp: int, num_sparsity_levels: int):
        for handle_name, handle in handles.items():
            if any(proj in handle_name for proj in ("self_attn.o_proj", "self_attn.out_proj", "mlp.down_proj")):
                
                if self.verbose:
                    dist_utils.print_on_main(f"Pruning {handle_name}")
                
                if any(proj in handle_name for proj in ("self_attn.o_proj", "self_attn.out_proj")):
                    expected_sparsity = mha_sparsity[block_id]
                    min_level = min(int(mha_sparsity[block_id] // (weights_diff_mha / handle.W.numel())), int(num_sparsity_levels // 2))
                    max_level = min(int((1 - mha_sparsity[block_id]) // (weights_diff_mha / handle.W.numel())), int(num_sparsity_levels // 2))
                    sparsities = [mha_sparsity[block_id] + l * weights_diff_mha / handle.W.numel() for l in range(-min_level, max_level + 1)]
                else: # any(proj in handle_name for proj in ("mlp.down_proj", )):
                    expected_sparsity = ffn_sparsity[block_id]
                    min_level = min(int(ffn_sparsity[block_id] // (weights_diff_mlp / handle.W.numel())), int(num_sparsity_levels // 2))
                    max_level = min(int((1 - ffn_sparsity[block_id]) // (weights_diff_mlp / handle.W.numel())), int(num_sparsity_levels // 2))
                    sparsities = [ffn_sparsity[block_id] + l * weights_diff_mlp / handle.W.numel() for l in range(-min_level, max_level + 1)]
                
                sparse_weights = handle.prune(sparsities, expected_sparsity, self.error_accumulation)
                if dist_utils.is_main():
                    for level, sparse_weight in enumerate(sparse_weights, start=-min_level):
                        os.makedirs(os.path.join(self.save_dir, handle_name), exist_ok=True)
                        # Map tensor to CPU before saving
                        torch.save(sparse_weight.cpu(), os.path.join(self.save_dir, handle_name, f"{level}.pth"))
                if dist_utils.is_dist_available_and_initialized():
                    dist.barrier()
                handle.reset()
