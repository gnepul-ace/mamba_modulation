# Mamba Modulation

## Introduction

This is the official repository for the supporting code to our NeurIPS 2025 paper [__Mamba Modulation: On the Length Generalization of Mamba Models__](https://openreview.net/forum?id=QEU047bE8p).

This repository is built off of [`MambaExtend`](https://github.com/ArminAzizi98/LongContextMamba). To use these codes properly, simply clone `MambaExtend` and then copy the files from the model you wish to run (`Mamba` or `Mamba2`)

We separate each of these their own separate folder due to changes in both the `modeling` files as well as the training/calibration/evaluation scripts.

We recommend running models on a machine with at least 12 GB of GPU memory with Ampere microarchitecture (our experiments have been tested on a 12GB GeForce RTX 4080 card).

### Differences with `MambaExtend`

Our primary changes relative to the `MambaExtend` repository is the modification of the scaling parameters. We provide two separate directories (`Mamba` and `Mamba2`) as the architectural differences requires the scaling to be applied differently.

### Running additional Baselines

To run additional baselines that were included in our paper, please refer to their individual repositories. For `DeciMamba`, we directly use checkpoints provided in [the official repository](https://github.com/assafbk/DeciMamba). For `LongMamba`, we use their checkpoints from [the official repository](https://github.com/jzhang38/LongMamba) as well. Note that because these methods are only applicable to one generation of the Mamba model; please refer to [our paper](https://openreview.net/forum?id=QEU047bE8p) for more explicit details.

## Citation

If you find our work helpful or interesting, please do not forget to cite our work as follows:

```
@inproceedings{lu2025mamba,
      title={Mamba Modulation: On the Length Generalization of Mamba Models},
      author={Peng Lu and Jerry Huang and Qiuhao Zeng and Xinyu Wang and Boxing Chen and Philippe Langlais and Yufei Cui},
      booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
      year={2025},
      url={https://openreview.net/forum?id=QEU047bE8p}
}
```
