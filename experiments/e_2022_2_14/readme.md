Yesterday's experiment produced pretty good reconstructions, with a couple of problems:

- missing high-end information
- 10-20hz "beating", especially in 4096 and 8192 channels, caused by oscillators

My, admittedly kind of silly hypothesis today is that, since the noise model's 
frequency resolution is too good, the oscillators are trying to fill the gaps 
that a coarser-frequency-resolution noise model would have filled.  The oscillators
perform poorly at this, "waving" wildly and producing a lot of "beating" and interference.

Today, I'll try a model with half the oscillators as the 2-9 model and with noise models
with 8 bands each