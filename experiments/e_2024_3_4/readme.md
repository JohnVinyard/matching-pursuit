It's been very difficult to avoid redundant events in previous models.

In this experiment, we'll break the problem down further.

First, decompose the audio into a room vector and a number of independent audio channels.

Then, apply the model I've been developing to further decompose events using the audio model.

If this doesn't work, approach as a clustering problem.  

Sparse code (all-at-once), then assign each atom to an event cluster index.  All atoms should
reconstruct the audio well, and subsets of event clusters should fool a discriminator.  We can also
learn an autoencoder to encode and decode clusters of variable size (how does this work?)