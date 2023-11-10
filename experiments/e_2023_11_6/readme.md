First, a little foray into FIR-like delay approximation, and then trying
some artificial restrictions on the expressiveness of otherwise 
arbitrarily-complex resonance model

# Things Yet to Try
- sparsity penalty
- tuning according to musical scale
- MSE loss
- single channel loss with overshoot penalty


# Overshoot
We start with a positive-valued representation of the audio, sum together
all events, and subtract.

Then, adding in each event one at a time, we move it nearer to the residual.

The problem with this approach is that we also move nearer to the noise 
introduced in other events.

To address this, we'll alter the loss thusly:

1. sum together all events and subtract
1. separate the difference into positive (residual) and negative (overshoot) 
   components
1. One at a time, add events back to the positive (residual) portion
1. Separate this difference into positive (residual) and negative (overshoot) components
1. Move the event closer to the positive residual, while penalizing any overshoot
   

# Resonance