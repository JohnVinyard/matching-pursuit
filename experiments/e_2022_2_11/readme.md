Despite changes to not square the noise component, the model continues to use oscillators to simulate
noise.  This could be caused by any combination of a couple factors:

- the loss function does not distinguish between stochastic and dense, oscillator-simulated noise
- the decoder is unable to sub out noise for the dense, oscillator-produced noise

Another possible theory is that oscillators are too densely packed in the lowest band.  Perhaps
making this less dense, or linearly-spaced would help?

# Conclusion

No huge changes here.  Both the noise and harmonic components exhibit a "beating" which might
be caused by the windowing in the loss function?  Also, the noise took on too many of the 
harmonic components.