Autoencoders work, but with lots of annoying artifacts.  GANs seem to be unstable, or to fall into mode collapse.

Use contrastive learning to learn a good embedding, and learn a generator that will promote reciprocity.

MLP Mixer seems to be at least partially responsible for overly-rhythmic generations.

Steps:
- removing MLP Mixer from generator/decoder

Next Up:
- use angle and don't enforce std=1
- remove MLP mixer from encoder; use conv and pooling instead

It seems to be very difficult to get the embedding to have the properties I want.