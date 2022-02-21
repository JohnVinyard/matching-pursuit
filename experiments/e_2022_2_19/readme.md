Yesterday's experiment strongly suggests that an adversarial loss could help to prevent some of the more annoying
artifacts that come with a naive MSE loss in the PIF space for autoencoders.  Although reconstructions were smoother,
they were not very precise and had "tuning" that was very off.

Things I'll try today to mitigate this:

- add a matching criteria to discriminator (unmatched samples are also "fake")
- return more features from the individual branches
- 2d decoder and/or encoder