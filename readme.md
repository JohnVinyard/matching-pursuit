This repository contains research and experiments aimed at producing sparse, interpretable representations of audio.

# Pre-Requisites

## Environment File Template

AUDIO_PATH=
PORT=9999
IMPULSE_RESPONSE_PATH=

## MusicNet

The [MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA) should be downloaded and extracted to a location on your local machine.

You can then update the `AUDIO_PATH` environment variable to point to the `musicnet/train_data` directory, wherever that may be on your local machine.

## Room Impulse Responses

Room impulse responses to support convolution-based reverb can be [downloaded here](https://oramics.github.io/sampled/IR/Voxengo/).

You can then update the `IMPULSE_RESPONSE` environment variable to point at the directory on your local machine that contains the
impulse response audio files.