# Mamba Modulation

## Introduction

This is the official repository for the supporting code to our NeurIPS 2025 paper __Mamba Modulation: On the Length Generalization of Mamba Models__.

This repository is built off of [`MambaExtend`](https://github.com/ArminAzizi98/LongContextMamba). To use these codes properly, simply clone `MambaExtend` and then copy the files from the model you wish to run (`Mamba` or `Mamba2`)

We separate each of these their own separate folder due to changes in both the `modeling` files as well as the training/calibration/evaluation scripts.

We recommend running models on a machine with at least 12 GB of GPU memory with Ampere microarchitecture (our experiments have been tested on a 12GB GeForce RTX 4080 card).

## Citation

```
@inproceedings{lu2025mamba,
      title={Mamba Modulation: On the Length Generalization of Mamba Models},
      author={Peng Lu and Jerry Huang and Qiuhao Zeng and Xinyu Wang and Boxing Chen and Philippe Langlais and Yufei Cui},
      booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
      year={2025},
      url={https://openreview.net/forum?id=QEU047bE8p}
}
```
