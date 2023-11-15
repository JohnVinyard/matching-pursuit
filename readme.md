This repository contains research and experiments aimed at producing sparse, interpretable representations of audio.

# Current Research

My current research is focused on sparse, interpretable representations of audio.  Ideally, the sparsity and interpretability also
leads to representations that can be manipulated at a level of granularity musicians are accustomed to.

![Sparse Representation](https://matching-pursuit-repo-media.s3.amazonaws.com/sparse_audio_represenation.png)

## Other Areas of Interest

- simpler, linear sparse decompositions such as matching pursuit
- perceptually-motivated loss functions, inspired by [Mallat's scattering transform](https://arxiv.org/abs/1512.02125) and the [Auditory Image Model](https://code.soundsoftware.ac.uk/projects/aim).


# Getting Started

## Environment File Template

```bash
AUDIO_PATH=
PORT=9999
IMPULSE_RESPONSE_PATH=
S3_BUCKET=
```

## MusicNet

The [MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA) should be downloaded and extracted to a location on your local machine.

You can then update the `AUDIO_PATH` environment variable to point to the `musicnet/train_data` directory, wherever that may be on your local machine.

## Room Impulse Responses

Room impulse responses to support convolution-based reverb can be [downloaded here](https://oramics.github.io/sampled/IR/Voxengo/).

You can then update the `IMPULSE_RESPONSE` environment variable to point at the directory on your local machine that contains the
impulse response audio files.

## My Trained Models

If you'd like to try out some of the models I've trained locally, you can set `S3_BUCKET` to `matching-pursuit-trained-models`