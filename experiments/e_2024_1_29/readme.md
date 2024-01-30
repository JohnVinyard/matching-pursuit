Not adding the context vector to event one-hot embeddings seems to help
with the diversity of codes.  Hopefully this leads to:

- a model that is forced to learn longer, more descriptive events, since there is a limited supply
- events/codes that are much more interpretible
- fewer redundant events;  since events must be longer and more descriptive, having multiple redundant events will push loss up

Things to try next:
- original single-channel loss
- normalization that never has across-sample dependencies