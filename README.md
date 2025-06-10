# Mamba Modulation

This repository contains our code for the paper "Mamba Modulation: On the Length Generalization of Mamba". The majority of the code is derived from [MambaExtend](https://github.com/ArminAzizi98/LongContextMamba).

## Setting up environments

In general, it is recommended to install the `requirements.txt` file by running
```bash
pip install -r requirements.txt
```
as this ensures that the versions of HuggingFace matches what is used for the models.

Occasionally, there may be an issue of `torch` not being installed, as it is a dependecy for `flash-attn`. In this case, we recommend installing
```bash
pip install torch torchvision torchaudio
```
and then running the first command again. If issues persist with symbols not being recognized, you may need to install `flash-attn` from a wheel directly.

**Note**: In our environment, we ran with `torch==2.6.0` and `flash-attn==2.6.3`.

## Running Code

For running code, please simply follow instructions from [MambaExtend](https://github.com/ArminAzizi98/LongContextMamba).

## Citation

Coming soon.
