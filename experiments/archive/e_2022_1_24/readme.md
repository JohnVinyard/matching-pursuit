Previous day's experiment went quite well, with the best autoencoding behavior thus far.
This experiment will begin the process of tweaking and improving on that one, with the first
step being to ensure that the oscillators are being employed and the good results aren't simply
the result of fitting noise.

# Findings

After forcing the noise model to have much better time resolution 
(and hence coarser time resolution) it became clear that the oscillator bank was contributing
nothing to the decoded result.  Tweaking and overfitting experiments do seem to indicate that
constraining each filter does produce output that includes the oscillator bank.