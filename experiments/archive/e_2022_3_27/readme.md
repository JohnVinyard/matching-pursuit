Today, I'm trying a single-band, oscillator+noise based generator and a discriminator that performs the entire PIF transform 
(including multi-band decomposition) as part of its forward pass.

Current issue: mode collapse

    Things to try:
    - batch norm
    - batch discrimination
    - gradient clipping
    - classic loss (non least-squares)
    - embedding task for discriminator
    - latent loss for discriminator
    


- disc latent vector similarity matrix - doesn't work