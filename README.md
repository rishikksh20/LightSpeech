# LightSpeech
UnOfficial PyTorch implementation of [LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search](https://arxiv.org/pdf/2102.04040). This repo uses the FastSpeech 2 implementation of Espnet as a base. This repo only implements the final version of LightSpeech model not the Neural Architecture Search as mentioned in paper.
But I am able to compress only 3x (from 27 M to 7.99 M trainable parameters) not 15x.


## Requirements :
All code written in `Python 3.6.2` .
* Install Pytorch
> Before installing pytorch please check your Cuda version by running following command : 
`nvcc --version`
```
pip install torch torchvision
```
In this repo I have used Pytorch 1.6.0 for `torch.bucketize` feature which is not present in previous versions of PyTorch.


* Installing other requirements :
```
pip install -r requirements.txt
```

* To use Tensorboard install `tensorboard version 1.14.0` seperatly with supported `tensorflow (1.14.0)`



## For Preprocessing :

`filelists` folder contains MFA (Motreal Force aligner) processed LJSpeech dataset files so you don't need to align text with audio (for extract duration) for LJSpeech dataset.
For other dataset follow instruction [here](https://github.com/ivanvovk/DurIAN#6-how-to-align-your-own-data). For other pre-processing run following command :
```
python .\nvidia_preprocessing.py -d path_of_wavs -c configs/default.yaml
```
For finding the min and max of F0 and Energy
```buildoutcfg
python .\compute_statistics.py
```
Update the following in `hparams.py` by min and max of F0 and Energy
```
p_min = Min F0/pitch
p_max = Max F0
e_min = Min energy
e_max = Max energy
```

## For training
```
 python train_lightspeech.py --outdir etc -c configs/default.yaml -n "name"
```

## For inference 
WIP
```
python .\inference.py -c .\configs\default.yaml -p .\checkpoints\first_1\xyz.pyt --out output --text "ModuleList can be indexed like a regular Python list but modules it contains are properly registered."
```
## For TorchScript Export
```commandline
python export_torchscript.py -c configs/default.yaml -n fastspeech_scrip --outdir etc
```
## Checkpoint and samples:
WIP


## References
- [LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search](https://arxiv.org/pdf/2102.04040)
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
- [ESPnet](https://github.com/espnet/espnet)
- [NVIDIA's WaveGlow implementation](https://github.com/NVIDIA/waveglow)
- [MelGAN](https://github.com/seungwonpark/melgan)
- [DurIAN](https://github.com/ivanvovk/DurIAN)
- [FastSpeech2 Tensorflow Implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [Other PyTorch FastSpeech 2 Implementation](https://github.com/ming024/FastSpeech2)
- [WaveRNN](https://github.com/fatchord/WaveRNN)
