First, a little foray into FIR-like delay approximation, and then trying
some artificial restrictions on the expressiveness of otherwise 
arbitrarily-complex resonance model

# Things Yet to Try
- ~~there is currently no path for the resonance without being convolved with the impulse!~~
- try delay-like resonances again if still getting detuned piano sound
- consider a single, time-varying coarse-grained frequency-domain filter applied _after_ the mixture
- ~~more complex/varied waveforms~~
- ~~log cumsum for more dynamic resonance/damping~~
- ~~coarse-grained, time-varying frequency domain filter (start with a single/static frequency-domain filter)~~
   - **This one makes and immediately noticeable difference**: also after limiting the frame-rate of the mixture
- ~~resonance should be applied _after_ the mixture, not independently to each~~


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