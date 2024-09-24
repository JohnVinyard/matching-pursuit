

This repository contains code, experiments and writing aimed at developing a
perceptually-lossless audio codec that is sparse, interpretable, and easy to manipulate. 
The work is currently focused on "natural" sounds, and leverages
knowledge about physics, acoustics, and human perception to remove 
perceptually-irrelevant or redundant information.

The basis of modern-day, lossy audio codecs is a model that slices audio into
fixed-size and fixed-rate "frames" or "windows" of audio.  This is even true of 
cutting-edge , ["neural" audio codecs such as Descript's Audio Codec](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419in%20a891d6591a68f814d5). 
While convenient, flexible, and generally able to represent the universe of possible sounds, it isn't easy to 
understand or manipulate, and is not the way humans conceptualize sound.  While modern music can sometimes deviate
from this model, humans in natural settings typically perceive sound as the combination of various
streams or sound sources sharing some physical space.


# Table of Contents

- [Recent Work](#recent-work)
  - [Sparse Interpretable Audio Model](#sparse-intereptable-audio-model)
  - [Gamma/Gaussian Splatting for Audio](#gammagaussian-splatting-for-audio)
  - [Overfitting as Encoder](#overfitting-as-encoder)
  - [Playable State-Space Model From Single Audio Example](#playable-state-space-model-from-single-audio-example)
  - [Other Areas of Interest](#other-areas-of-interest)
- [Getting Started](#getting-started)
  - [Environment File Template](#environment-file-template)
  - [MusicNet](#musicnet)
  - [Room Impulse Responses](#room-impulse-responses)



# Recent Work

![sparse decomposition of audio](https://zounds-blog-media.s3.amazonaws.com/sparse_audio.png)

## Sparse Intereptable Audio Model

This [small model attempts to decompose audio](https://johnvinyard.github.io/siam.html) featuring acoustic instruments
into the following components:

Some maximum number of small (16-dimensional) event vectors, representing individual audio events
Times at which each event occurs.

![Sparse Interpretable Audio Model](https://matching-pursuit-repo-media.s3.amazonaws.com/vector_siam.drawio2.png)

## Gamma/Gaussian Splatting for Audio

[In this work, we apply a Gaussian Splatting-like approach to audio](https://johnvinyard.github.io/gamma-audio-splat.html)
to produce a lossy, sparse, interpretable, and manipulatable representation of audio. We use a source-excitation model
for each audio "atom" implemented by convolving a burst of band-limited noise with a variable-length "resonance", which
is built using a number of exponentially decaying harmonics, meant to mimic the resonance of physical objects. Envelopes
are built in both the time and frequency domain using gamma and/or gaussian distributions. Sixty-four atoms are randomly
initialized and then fitted (3000 iterations) to a short segment of audio via a loss using multiple STFT resolutions. A
sparse solution, with few active atoms is encouraged by a second, weighted loss term. Complete code for the experiment
can be found on GitHub. Trained segments come from the MusicNet dataset.


## Overfitting as Encoder

Assuming that our _decoder_ is parameter-less (i.e., only the encoded representation is needed), 
and that the path from decoder to loss function is fully differentiable, then we
can perform gradient descent to find the best encoded representation for a given
piece of audio.  The [Gaussian/Gamma Splatting Experiment](https://johnvinyard.github.io/gamma-audio-splat.html)
is an example of this;  audio is defined as a set of events/parameters passed to
a fixed synthesizer.

## Playable State-Space Model From Single Audio Example

[This work](https://blog.cochlea.xyz/ssm.html) takes a slightly different approach to extracting
a sparse audio representation, but ties in to the [Overfitting as Encoder](#overfitting-as-encoder)
idea.  We attempt to decompose a single musical audio signal into two distinct components:

- A state-space model representing the resonances, or transfer function of the system as whole, 
   including the instrument and the room in which it was played
- A sparse control signal, representing the ways in which energy is injected into the system


## Other Areas of Interest

- simpler, linear sparse decompositions such as matching pursuit
- perceptually-motivated loss functions, inspired by [Mallat's scattering transform](https://arxiv.org/abs/1512.02125)
  and the [Auditory Image Model](https://code.soundsoftware.ac.uk/projects/aim).
- approximate convolutions for long kernels
  - since many useful kernels in music are highly redundant, can we find a 
    low-rank approximation in the frequency domain and perform the multiplication/convolution there?
  - approximate convolutions via [hyperdimensional computing](https://www.dafx.de/paper-archive/details/CsiwHdDxhJGtNCojXPixcQ)


# Getting Started

## Environment File Template

```bash
AUDIO_PATH=
PORT=9999
IMPULSE_RESPONSE_PATH=
S3_BUCKET=
```

## MusicNet

The [MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA) should be downloaded and extracted to a location
on your local machine.

You can then update the `AUDIO_PATH` environment variable to point to the `musicnet/train_data` directory, wherever that
may be on your local machine.

## Room Impulse Responses

Room impulse responses to support convolution-based reverb can
be [downloaded here](https://oramics.github.io/sampled/IR/Voxengo/).

You can then update the `IMPULSE_RESPONSE` environment variable to point at the directory on your local machine that
contains the
impulse response audio files.