Early experiments using a randomly-initialized dictionary show some promise.

I should probably do things in the following order:
    - learn an autoencoder for the pif feature to reduce dimensionality
    - learn a sparse dictionary for the compressed pif feature
    - learn a network that produces audio from either of the prior two representations