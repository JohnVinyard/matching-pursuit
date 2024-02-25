Try with an explicit switching mechanism to turn events on or off

# In Progress
- discriminator with masking to ensure event-independence

# Outcomes


# Things to Add
- add a loss to ensure that certain times/frames aren't over-represented;  distribution should be random
- event vectors could be quantized in a compression phase
- remove amplitude to simplify model
- l0 norm for sparsity
- would softmax work better than a scalar for the event switching?
- more efficient way to split each non-zero value into its own channel

- fft_shift or some other fine-grained scheduling adjustment
- padding at the beginning to allow events to start before the window the model can see
- different versions of covariance loss?