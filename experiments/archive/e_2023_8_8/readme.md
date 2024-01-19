# Goals

- avoid convolutions with excess padding
- sparse rather than dense generation

# Model Summary

## Encoder

1. gather context
2. sparsify
3. generate dictionary from 

## Decoder


## Things to Try

- test with `soft=True` for sparsify - very early results seem to suggest it might
- more complex decoder
- per-event output from sparsify + condensed events