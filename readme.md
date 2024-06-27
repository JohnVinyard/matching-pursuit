This repository contains research and experiments aimed at producing sparse, interpretable representations of audio.

# Current Research

My current research is focused on sparse, interpretable representations of audio.  Ideally, the sparsity and interpretability also
leads to representations that can be manipulated at a level of granularity musicians are accustomed to.

![Sparse Representation](https://matching-pursuit-repo-media.s3.amazonaws.com/sparse_audio_represenation.png)

## Sparse Intereptable Audio Model

This [small model attempts to decompose audio](https://johnvinyard.github.io/siam.html) featuring acoustic instruments into the following components:

Some maximum number of small (16-dimensional) event vectors, representing individual audio events
Times at which each event occurs.

![Sparse Interpretable Audio Model](https://matching-pursuit-repo-media.s3.amazonaws.com/vector_siam.drawio2.png)

## Gamma/Gaussian Splatting for Audio

[In this work, we apply a Gaussian Splatting-like approach to audio](https://johnvinyard.github.io/gamma-audio-splat.html) to produce a lossy, sparse, interpretable, and manipulatable representation of audio. We use a source-excitation model for each audio "atom" implemented by convolving a burst of band-limited noise with a variable-length "resonance", which is built using a number of exponentially decaying harmonics, meant to mimic the resonance of physical objects. Envelopes are built in both the time and frequency domain using gamma and/or gaussian distributions. Sixty-four atoms are randomly initialized and then fitted (3000 iterations) to a short segment of audio via a loss using multiple STFT resolutions. A sparse solution, with few active atoms is encouraged by a second, weighted loss term. Complete code for the experiment can be found on github. Trained segments come from the MusicNet dataset.


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