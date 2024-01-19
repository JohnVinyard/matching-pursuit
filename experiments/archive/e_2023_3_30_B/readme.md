TODO: 

- for differentiable index experiment, also try a loss that compares the self/self matrix to the self/other matrix
    - maybe this is an alternative (or equivalent?) way to get gradients to flow?
- filter amplitudes could be applied to a multi-band decomposition, to allow for well-tuned lower frequencies
- apply the new technique to get a differentiable matching pursuit algorithm that can be used as a loss
- compute embeddings for atoms
- positional encoding math investigation (e.g, does P1 + P6 = P7?, if not, which is it closest to?)
- try applying differentiable matching pursuit loss to my physics-based experiments
- given a 2D time/atom feature map, establish a technique to obtain
    - an amplitude (easy)
    - a time scalar (figured this out today)
    - an atom embedding (just dot product with all atom embeddings)
    - ... in a fully differentiable way
