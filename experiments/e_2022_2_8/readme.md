It seems that any kind of adversarial loss is really hindering the model's ability to learn, but yesterday's experiment
also revealed that different activations for amplitude and frequency are very helpful.

Today's experiment will be a vanilla autoencoder one, leveraging the activation findings from yesterday.

# Conclusion

This decoder actually makes use of the osciallators and learns some harmonic structures.  There's some beating or artifacts
caused (maybe) by densely spaced oscillators in the lower bands.