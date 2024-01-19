What tweaks can I make to yesterday's vanilla autoencoder experiment to make it work better?

Does the experiment work better overall with log-spaced constrained frequencies?

# Conclusions

This experiment has worked *very* well, both in terms of reconstruction quality and in
terms of a latent space that looks very promising.  The main issue is that the noise
model seems to have faded into silence, leaving clusters of oscillators to try and 
reproduce wider-band noise, which is very perceptually different.  Final losses were in the 
7.5 - 10 range.