Can I directly optimize (i.e., backpropagate all the way through) my synthesizer model?

It's OK if not;  the real plan here is train a neural network to approximate the loss
from the synth parameters and optimize those directly, since it seems to be pretty
difficult to optimize all the way back through audio, oscillators, etc.