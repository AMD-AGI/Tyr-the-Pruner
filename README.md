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
