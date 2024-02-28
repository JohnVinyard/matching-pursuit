Current theory:  Models are missing notes because the safest strategy with the discriminator is to
produce just one or two well.


We don't just have a single loss function, we have tens of them, at least, and they vary over time.  We
have to learn to balance them such that none is ever too low


Try with an explicit switching mechanism to turn events on or off

# Benefits of this Model
- streaming, no fixed size

# In Progress
- l0 norm for sparsity.  Will this obviate the need for shift loss?

# Next Up
- shift loss to ensure that the network isn't learning fixed event positions
- padding at the beginning to allow events to start before the window the model can see
- add a loss to ensure that certain times/frames aren't over-represented;  distribution should be random (what is the likelihood/expectation of an event at a particular position?)

# Outcomes
- discriminator loss with masking helps! Keep it


# Things to Add
- more efficient way to split each non-zero value into its own channel
- event vectors could be quantized in a compression phase
- remove amplitude to simplify model
- would softmax work better than a scalar for the event switching?
- fft_shift or some other fine-grained scheduling adjustment
- different versions of covariance loss (with subsampling)?
- would residual covariance loss support more crisp events?
- remove hard-coded number of frames, replace with a ratio of input size
- there needs to be a replinishing/leaking bucket budget for events, rather than the fixed number of events now.




# Questions
- how do I correct the bias toward earlier events?  Since we're computing the loss over the entire
  segment, regardless of an event's support/domain, there's an inherent bias toward earlier events