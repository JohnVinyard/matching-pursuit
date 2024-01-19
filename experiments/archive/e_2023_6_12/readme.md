Early experiments using a randomly-initialized dictionary show some promise.

I should probably do things in the following order:
    - learn an autoencoder for the pif feature to reduce dimensionality
    - learn a sparse dictionary for the compressed pif feature
    - learn a network that produces audio from either of the prior two representations

# Things to try
- refractory period
- fixed atoms learned by matching pursuit (the problem here is that we won't learn good noise representations)


If pif is (batch, freq_bands, time, periodicity), maybe I should be working toward
a sparse autoencoder with atoms like (1, freq_bands, SHORTER, periodicity)

# Findings
- the greater the norm of the atoms, the slower they are to change, and the sparser
  the control signal becomes