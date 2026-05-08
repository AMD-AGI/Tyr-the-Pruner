[![License](https://img.shields.io/github/license/AMD-AGI/Tyr-the-Pruner.svg?style=flat)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/AMD-AGI/Tyr-the-Pruner.svg?style=flat)](https://github.com/AMD-AGI/Tyr-the-Pruner/graphs/contributors)

# Týr-the-Pruner

> Structural pruning for LLMs via global sparsity distribution optimization.

Týr-the-Pruner is a research codebase for structural pruning of large language models. The project implements global sparsity distribution optimization to search for effective pruning configurations across transformer layers and modules. It is intended for AMD research use and supports experiments around LLM compression, sparsity, and inference efficiency.

Paper: [Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization](https://arxiv.org/abs/2503.09657)

## Setup

The reference environment used for experiments is listed below:

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

Run the example script:

```bash
bash script/example.sh
```

The example performs iterative supernet pruning followed by sparsity distribution search for Llama-2-7b-hf across multiple sparsity levels.

Before running, update cache paths and model access settings in the script as needed:

```bash
export HF_DATASETS_CACHE=/path/to/ample/storage/space
export TRANSFORMERS_CACHE=/path/to/ample/storage/space
```

## Contact

For questions, issues, or contributions, please reach out to the maintainers:

- Guanchen Li — [@guanchenl](https://github.com/guanchenl) · guanchen.li@amd.com

> Note: For internal or private AMD repositories, maintainers must list their AMD email address.

See [CODEOWNERS](.github/CODEOWNERS) for the full ownership list.

## Acknowledgements

This project makes use of open-source code and ideas from the following repositories:

- [EvoPress](https://github.com/IST-DASLab/EvoPress): Evolutionary structured pruning and compression for transformer models.
- [FLAP](https://github.com/CASIA-IVA-Lab/FLAP): Fast Layer-wise Adaptive Pruning framework for efficient deep networks.
- [OSSCAR](https://github.com/mazumder-lab/OSSCAR): Open-source framework for sparse compression and acceleration of deep neural networks.

We sincerely thank the authors and contributors of these excellent works for making their code publicly available.

## Citation

```bibtex
@inproceedings{Li2025TyrThePruner,
  title        = {Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization},
  author       = {Li, Guanchen and Xu, Yixing and Li, Zeping and Liu, Ji and Yin, Xuanwu and Li, Dong and Barsoum, Emad},
  booktitle    = {NeurIPS 2025},
  year         = {2025},
  url          = {https://neurips.cc/virtual/2025/poster/115807}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
