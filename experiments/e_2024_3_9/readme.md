Try the same model I've been having some success with, but with an iterative approach to decomposition, 
rather than all-at-once

# TODO

- if the l1 norm loss doesn't work to impose sparsity, go back to using `detach()` in `single_channel_loss`
  to make gradients of individual channels independent

  # In the Next Experiment
  - lose the context vector entirely;  everything depends on the event vector
  - use audio buffers rather than audio elements to play individual sounds
  - bump up sparsity penalty a bit
  - show low-res view of audio "channels", not just event positions
  - update architecture diagram
  - try out info/entropy-based loss again and see where it gets us
  - try out other resonance model, just for fun