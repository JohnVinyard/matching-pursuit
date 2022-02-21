Yesterday's experiment showed that an adversarial loss (vs a hand-designed MSE loss in PIF space) does significantly
cut down on beating and other annoying artifacts, but the generator did not learn to reconstruct samples, leading
me to think that the discriminator was ignoring the conditioning information.

Now, I ask it to judge along two dimensions, realness and "matched-ness"