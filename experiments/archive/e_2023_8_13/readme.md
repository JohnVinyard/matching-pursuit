Are there certain ways to express fundamental frequency that lead to better optimization?  This could help to better design an experiment that uses
synthesizers to "explain" an audio segment


There are two aspects to explore...

## Synth Models
- scale values for f0
- binary representation of f0
- one of N (less flexible, powerful)
- wavetable (how does this work?)
    - scale read positions are turned into normal sample distributions and then used to read from the wavetable

## Losses
- per-sample MSE loss
- STFT loss (with phase)
- STFT loss (without phase)
- FFT loss (entire signal, with phase)
- FFT loss (entire signal, without phase)