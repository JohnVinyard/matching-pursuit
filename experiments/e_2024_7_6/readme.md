Hybrid of the last two major experiments, which were:
    - autoencoder that derives a sparse representation
    - overfitting gaussian-splatting-like model


This experiment will use the same encoder as the first experiment, but a completely parameter-free decoder from the second.

# Conclusion

While I'm sure this is _possible_, the gaussian splatting experiment required a lot of hand-tuning
of the initial parameters.  Using gaussian and gamma distributions for several parameters seems to
lead to a _very_ spiky, hard-to-optimize landscape.