# Sparse Interpretible Audio Model Loss

## Desiderata

The model should produce a sparse event-based representation of an audio segment.  
Each segment should be explained using as few atoms/events as possible, and only
a single atom/event should be required to reproduce a single, real-world event, e.g.
a single piano key being struck, or a legato violin note.  The model and its loss should
be less focused on an _exact_ (i.e. residual with 0 norm) loss, and more on a loss that
removes as much _information_ as possible from the signal, such that the residual is
indistinguishable from random noise.

### A Note on Sparsity Losses

While the model should be as sparse as possible in its explanation, I'm hestitant to use an l1
or l0 norm loss directly, since it requires fiddly and seemingly arbitrary balancing of the sparsity
and other losses.  It's incredibly easy for the model to simply output silence and get a pretty good score.

In the "Spiking Music" paper, they use an annealing schedule, starting with a sparsity loss coefficient of 0,
and increasing it twice throughout training.

My approach thus far has been to approach sparsity indirectly, either via encouraging orthogonal events,
or by encouraging competition amongst events.


## Current Tries, and Their Problems

### Randomized Channel Optimization

The first try works as follows.  At each iteration, remove the sum of all channels from the input
segment.  Then, pick a channel at random and ask it to explain more of the existing residual by pulling
it in that direction.  This loss has no sparsity objective, no global loss, and has the potential drawback
that it is slowly pulling each event in a random direction, making them noisier and fuzzier, and hence
less representative of real-world events.  That said, this continues to be the best performing loss, which
is a complete mystery to me.

It was inspired by the observation that, in matching pursuit, each channel is optimized independently, and
in sequence, such that no two atoms/events are ever asked to explain the same components of the signal.  My hope
was that, by optimizing a single channel at a time, the events would be pulled in opposing directions and become
more orthogonal.

### Channel Closest to Residual Optimization

This idea is almost the same as "Randomized Channel Optimization", but tries to address the potential problem of
channels slowly getting fuzzy/averaged/washed-out by always choosing the channel closest to the residual.  It seems
to work similarly, but again, has no global loss or sparsity penalty.

### Loudest to Quietest Optimization

This loss first sorts each channel by its norm, desending (most-to-least energy) and then removes them from the original
signal, on at a time.  At each step, we try to maximize the reduction in norm.  Note that if a channel "overshoots", this
is increases the norm.  The idea here was to introduce some competition between channels, hence, indirectly encouraging sparsity.
Also, there is an implicit global loss, since we try to maximize the reduction in norm.  If the norm is 0 at a given step, the
only non-penalized choice is to be completely silent.

The problem with this approach is that asking each event to explain _as much of the residual as possible_ isn't quite right.
There's a point where the event is "perfect", given our desiderata, and this loss continues to ask it to explain more.  This
was apparent in listening tests;  sparsity was induced (I think?), but it was clear that the few events present were fuzzy and
attempting to explain too much of the signal.  _How do we know when to stop_, given this approach?

### Orthogonal Loss

In this approach, we use a standard, global penalty on the l1 or l2 norm of the residual, but also look at the correlation between
each channel pair.  Ideally, each channel is completely independent (not overlapping in time or frequency content) and my thought
was that this would also encourage sparsity, since the easiest way to be orthogonal, uncorrelated is to be completely silent.

One failure mode of orthogonal loss is that the easiest (most pathological) way to reduce this loss is to separate each event
in _time_ only, such that each event represents a short, full-band snippet of audio.

> TODO: One possible "fix" here would be to first blur in the time dimension before finding the pairwise correlations

### Patch Info Loss

In this approach, we divide input and reconstruction spectograms into overlapping patches, give each unit norm, and
train an auxillary vector quantizer to assign each patch a discrete code.  We then use the inverse frequency within
the batch as a proxy for the amount of "information" contained in each patch, and hence, the weight/importance of the
delta between the real and reconstructed patch.  There is no push for sparsity in this approach, but, the weighting did
clearly cause the model to focus more on perceptually relevant details, instead of the blurry result we generally get from
plain l1 or l2 residual norm loss.

The _problem_ here mainly arose from boundary artifacts, there was always unpleasant "beating" that aligned with patch
boundaries.  Also, the VQ model was memory intensive, due to the number of patches for each spectrogram.

> TODO: investigate approaches that will remove the boundary artifacts, and make the VQ model more energy efficient.

### Neural Matching Pursuit Loss

This approach also tries to emulate matching pursuit benefits.  We sort the channels in descending order by some
criteria (e.g., loudness, max value, correlation with position, etc.).  We choose a target for that channel by
averaging together the channel alone with the channel + residual. The loss is the sum of distances at each step.
The key difference here is that _we remove this target, and not the channel_ from the residual, as if the channel
were already doing better than it is.  This is meant to guard against multiple channels trying to explain the same
parts of the audio.

The first attempt at this was a total failure, possibly because of the problem just mentioned.  It may not be feasible to
be without some kind of global loss


## Problems

- I think some sort of global loss will utimately be required, but global losses generally lead to fuzzy results
- While the patch info loss has worked well, there are always boundary artifacts
- Many of the step-by-step approaches make it possible that subsequent channels are accounting for noise introduced by earlier channels


## Future Directions

### Noise Distribution Loss

Rather than seeking to reduce the residual to 0, let's say our global objective is to make the residual as close to 
a uniform/normal distribution as possible.  This could be a step-by-step process as well.

### Step-by-Step Patch Info Loss

compute the total information of a segment of audio by doing `sum([1 / sum(patch) for patch in unique_patches])`.  At each step,
we seek to reduce the total information in that segment as much as possible.  This is related to the noise distribution loss.

Taking this approach would require a much more efficient VQ model, however.

### Amplitude Distribution Loss

Take the KL-divergence between desired and actual amplitude distributions.  Again, this requires that an arbitrary/balanced target
distribution and weighting.


# Things to Try

Everything is stochastic:
- the model learns to be as sparse as possible, because drop random numbers of the bottom N events
- a center of gravity for the event is computed, and it is asked to explain the segment within the top 90th percentile of that coverage/support
- New physical constraint: model has learnable resonances, but must pick exactly one.



decompose loss using matching pursuit and assign different losses to each channel

finally, any remaining residual loss belongs to entire network.


- l0 + most-highly-correlated channel (with detach())
    - very choppy audio
    - weird strobing effect
- l0 + most-highly-correlated channel (without detach())
- l0 + closest channel (with detach())
- l0 + closest channel (without detach())
- l0 + random_channel (with mask and detach())
- l0 + random_channel (with mask and without detach())
- l0 + global transform l1 loss



# Questions

If "single channel loss" was always optimiizing over all channels, why did this generally 
produce more crisp reconstructions?  Or did it, was that an illusion?  When I use `detach()`,
am I _actually_ seeing the "optimizing other channels' noise" issue for the first time?

Does use the variant of single_channel_loss where we choose the closest channel, rather than
a random one, produce events with less of a noisy background/tail?