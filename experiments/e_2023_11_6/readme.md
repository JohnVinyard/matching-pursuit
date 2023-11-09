First, a little foray into FIR-like delay approximation, and then trying
some artificial restrictions on the expressiveness of otherwise 
arbitrarily-complex resonance model

# Things Yet to Try
- ~~try delay-like resonances again if still getting detuned piano sound~~
   - while slower to train, it results in less noise/aliasing and sounds nicer overall
- consider a single, time-varying coarse-grained frequency-domain filter applied _after_ the mixture.  
   If we think of this resonance as the instrument body, it makes sense that it would change shape
   uniformly, and not vary wildly as the string or tube changes shape


# Overshoot
We start with a positive-valued representation of the audio, sum together
all events, and subtract.

Then, adding in each event one at a time, we move it nearer to the residual.

The problem with this approach is that we also move nearer to the noise 
introduced in other events.

To address this, we'll alter the loss thusly:

1. sum together all events and subtract
1. separate the difference into positive (residual) and negative (overshoot) 
   components
1. One at a time, add events back to the positive (residual) portion
1. Separate this difference into positive (residual) and negative (overshoot) components
1. Move the event closer to the positive residual, while penalizing any overshoot
   

# Resonance