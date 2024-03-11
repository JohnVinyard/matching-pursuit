Try the same model I've been having some success with, but with an iterative approach to decomposition, 
rather than all-at-once

# TODO

- if the l1 norm loss doesn't work to impose sparsity, go back to using `detach()` in `single_channel_loss`
  to make gradients of individual channels independent