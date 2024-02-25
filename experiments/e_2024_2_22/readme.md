Add in a discriminator loss, even though I've been trying to avoid it.

Instead of lowering the norm at each step, I should focus on _removing_ information
from one signal and _increasing_ information in the reconstruction.


I have to find a way to incorporate frequency pos encodings into the patches.
The problem I ran into was that by adding pos encodings and then projecting back
down, I ran the risk of model collapse, i.e., if I do this step before patch-ifying,
then it's possible for the model to just output the same patch for everything.

My noise generator is insufficient for certain types of noise.  Ideally, a scratchy
record sound in the background would be a single event, but this may spend the entire
event budget.



# Things to Try
- try disc + original single-channel loss (BEST SO FAR)
- try disc + single-channel loss B
- try without channel masking
- try with conditional discriminator (what is the conditioning?)


# Question
- why does single_channel_loss_3 + disc loss _start_ off so well, but then become less crisp


Complex-valued matching pursuit, learnable, complex-valued kernels

# Different approaches
- tension between descriptiveness and generalization via discrete set of possibilities
- physical, energy-based limitations on the amount of energy used and complexity of harmonic structure


Loss ideas:
- total info loss
- regardless of scale, minimize the mean or sum of the covariance matrix of the residual, in other words, push all
  information into reconstruction



How can I build an architecture that can:

- operate on any length of data?
- produce events of any length?

A NERF model holds promise here, if positional encodings are chosen wisely (look into ROPE embeddings)


What if I learned an ensemble of NERF models that all take the same input vector and add together at the end?
What would the different channels sound like?

What if I think of each pixel of a spectrogram as being explained by some number of events that precede that
pixel in time?