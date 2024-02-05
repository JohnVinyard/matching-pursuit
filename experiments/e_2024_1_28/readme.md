Get rid of up-scaling and sparsity, just generate events directly

Additional Changes/losses:
    - distinct event and time vectors
    - event vectors are are vector quantized
    - time vectors should come from normal distribution
    - context vectors should come from normal distribution
    - tau should be learnable
    - tau should be encouraged to be small
    - every nth iteration, a random pattern should be generated


## Analyzer

Should output a set of N event vectors (normal distribution) and a set of timings.
Regardless of gumbel or normal softmax, events should encouraged to be crisp/confident

## Random Generation

Generation consists of:
    - `N` normally distributed event vectors
    - `N` one-hot timining representations
