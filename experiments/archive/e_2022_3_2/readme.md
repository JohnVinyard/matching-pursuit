Try a vanilla GAN again, this time with convolutional disc and gen

I then tried a variant where an encoder also tries to predict the latent vector, and the
encoder and generator are optimized jointly to this end.  Then encoder ended up doing a fairly 
good job of predicting the latent vector, but generated samples still collapsed into a single
example, and gradients seem to have exploded