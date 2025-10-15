# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src import dist_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class LocalPruner:
    def __init__(
        self, 
        layer: nn.Module, 
        layername: str = None,
        num_heads: int = 32,
        mlp_update_iter: int = 16, 
        mha_update_iter: int = 1,
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = layer.weight.data.clone().t()
        self.d_row, self.d_col = layer.weight.shape
        self.layername = layername
        self.num_heads = num_heads

        # Local Pruning Hyperparameters
        self.mlp_update_iter = mlp_update_iter
        self.mha_update_iter = mha_update_iter

        # Layer Properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape

        # Initialize Hessian
        self.H = torch.zeros((self.d_col, self.d_col), device=self.W_device)
        self.nsamples0 = 0

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "LocalPruner supports only linear and convolutional layers."

    @torch.no_grad()
    def update(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        inp = inp.float()
        self.nsamples0 += tmp
        self.H += (inp).matmul(inp.t())

    @torch.no_grad()
    def reset(self) -> None:
        self.W = self.layer.weight.data.clone().t()
        self.H = torch.zeros((self.d_col, self.d_col), device=self.W_device)
        self.nsamples0 = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def step(self, B, sparsities: List[float], expected_sparsity, error_accumulation) -> List[Tensor]:
        device, dtype = self.W_device, self.W_dtype

        if dist_utils.is_main():
            torch.cuda.empty_cache()
            sparse_weights = []
            expected_activation = torch.zeros_like(B.t(), device=device, dtype=dtype)
            expected_sparsity = 0

            for i, sparsity in enumerate(sparsities):
                if "self_attn.out_proj" in self.layername or "self_attn.o_proj" in self.layername:
                    cin = self.num_heads
                    sp = sparsity
                    upd_iter = self.mha_update_iter
                else:
                    cin = B.shape[0]
                    sp = sparsity
                    upd_iter = self.mlp_update_iter
                
                if sparsity == 0.0:
                    B_sol, _ = B.clone(), 0
                elif sparsity == 1.0:
                    B_sol, _ = torch.zeros_like(B, device=device, dtype=dtype), float('inf')
                else:
                    B_sol, _ = self.local_prune_core(B.clone(), self.H, self.G, cin, int(cin * (1-sp)), upd_iter)
                dist_utils.print_on_main(f"Pruning {self.layername} with Sparsity {sparsity}; remain {int(cin * (1-sp))} / {cin}")
                # Append the current pruned weight
                sparse_weights.append(B_sol.t().contiguous().to(device=device, dtype=dtype))

                if error_accumulation and sparsity == expected_sparsity:
                    # Update: Median version (line 99-100) of weighted expectation error accumulation (line 101-105), faster
                    expected_activation = B_sol.t().contiguous().to(device=device, dtype=dtype)
                    self.layer.weight.data = expected_activation
            #     if error_accumulation:
            #         expected_activation += B_sol.t().contiguous().to(device=device, dtype=dtype) * (1 - sparsity)
            #         expected_sparsity += 1 - sparsity
            # if error_accumulation:
            #     self.layer.weight.data = expected_activation / expected_sparsity
        else:
            sparse_weights = [
                torch.empty_like(self.W.t(), device=device, dtype=dtype)
                for _ in sparsities
            ]

        # Synchronize pruned weights across distributed workers
        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()
            for i in range(len(sparsities)):
                dist.broadcast(sparse_weights[i].contiguous(), src=0)

        return sparse_weights

    def prune(self, sparsities: List[float], expected_sparsity, error_accumulation) -> List[Tensor]:
        W = self.layer.weight.data.clone()
        W = W.float()
        dead = torch.diag(self.H) == 0
        B = W.t()
        B[dead,:] = 0  
        self.H += torch.eye(B.shape[0]).to(self.W_device) * torch.mean(torch.diag(self.H)) * 1e-2
        self.G = (self.H @ B)
        sparse_weights = self.step(B, sparsities, expected_sparsity, error_accumulation)
        return sparse_weights

    @staticmethod
    def local_prune_core(W, H, G, num_total_groups: int, num_groups_to_remain: int, update_iter: int = 1):
        device = W.device
        cin, cout = W.shape
        group_size = int(cin / num_total_groups)

        H_inv = torch.linalg.inv(H)
        W_g = W.reshape(num_total_groups, group_size, cout)
        group_abs_sum = torch.sum(torch.abs(W_g), dim=(1, 2))
        pruned_group_mask = (group_abs_sum <= 1e-12)
        num_already_zero = int(pruned_group_mask.sum().item())

        if num_already_zero > 0:
            zero_idx = torch.cat([
                torch.arange(g * group_size, (g + 1) * group_size, device=device)
                for g in torch.nonzero(pruned_group_mask, as_tuple=False).flatten()
            ])
            H_inv[zero_idx, :] = 0
            H_inv[:, zero_idx] = 0
            W = H_inv @ G
            if (num_total_groups - num_groups_to_remain - num_already_zero) <= 0:
                W[zero_idx, :] = 0
                return W, 0.0

        remaining_to_prune = int(num_total_groups - num_groups_to_remain - int(pruned_group_mask.sum().item()))
        if remaining_to_prune <= 0:
            kept_mask = (~torch.repeat_interleave(pruned_group_mask, group_size))
            W_kept = torch.zeros_like(W)
            try:
                W_kept[kept_mask, :] = torch.linalg.inv(H[kept_mask][:, kept_mask]) @ G[kept_mask, :]
            except Exception:
                H_cpu = H[kept_mask][:, kept_mask].cpu()
                G_cpu = G[kept_mask, :].cpu()
                W_kept[kept_mask, :] = (torch.linalg.inv(H_cpu) @ G_cpu).to(device)
            prune_loss = torch.sum(-W_kept * G + 0.5 * W_kept * (H @ W_kept)).detach().item()
            return W_kept, prune_loss

        update_rounds = max(int(min(update_iter, remaining_to_prune)), 1)
        base, extra = divmod(remaining_to_prune, update_rounds)
        groups_to_prune_each_round = torch.full((update_rounds,), base, dtype=torch.int, device=device)
        if extra > 0:
            groups_to_prune_each_round[:extra] += 1

        for round_id in range(update_rounds):
            if group_size > 1:
                obj_mat = torch.zeros_like(W)
                for g in range(num_total_groups):
                    if pruned_group_mask[g]:
                        continue
                    sl = slice(g * group_size, (g + 1) * group_size)
                    H_block = torch.linalg.inv(H_inv[sl, sl])  # ≈ H[sl, sl]
                    obj_mat[sl, :] = (H_block @ W[sl, :] / 2.0) + G[sl, :]
            else:
                diag_Hinv = torch.diag(H_inv)
                safe_den = (pruned_group_mask.to(W.dtype) + diag_Hinv).clamp_min(1e-12)
                obj_mat = (1.0 / safe_den)[:, None] * (W / 2.0) + G

            obj_val = (W * obj_mat).reshape(num_total_groups, group_size, cout).sum(dim=(1, 2))
            obj_val_masked = obj_val + 1e20 * pruned_group_mask.to(obj_val.dtype)

            sorted_groups = torch.argsort(obj_val_masked)
            k = int(groups_to_prune_each_round[round_id].item())
            pick_groups = sorted_groups[:k]
            pick_idx = torch.cat([
                torch.arange(g * group_size, (g + 1) * group_size, device=device)
                for g in pick_groups
            ])

            Hinv_block_inv = torch.linalg.inv(H_inv[pick_idx][:, pick_idx])  # ≈ H[pick_idx, pick_idx]
            W -= H_inv[:, pick_idx] @ Hinv_block_inv @ W[pick_idx, :]
            W[pick_idx, :] = 0
            H_inv -= H_inv[:, pick_idx] @ Hinv_block_inv @ H_inv[pick_idx, :]
            H_inv[pick_idx, :] = 0
            H_inv[:, pick_idx] = 0
            pruned_group_mask[pick_groups] = True

        W_pruned = torch.zeros_like(W)
        kept_mask = (~torch.repeat_interleave(pruned_group_mask, repeats=group_size))
        try:
            W_pruned[kept_mask, :] = torch.linalg.inv(H[kept_mask][:, kept_mask]) @ G[kept_mask, :]
        except Exception:
            H_cpu = H[kept_mask][:, kept_mask].cpu()
            G_cpu = G[kept_mask, :].cpu()
            W_pruned[kept_mask, :] = (torch.linalg.inv(H_cpu) @ G_cpu).to(device)

        prune_loss = torch.sum(-W_pruned * G + 0.5 * W_pruned * (H @ W_pruned)).detach().item()
        return W_pruned, prune_loss
