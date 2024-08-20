Relax, and make things easier, at least to start.  The model can use up to 64 events
but there's an auxillary l0-norm loss to encourage the sparsest-possible solution

# To Try

- iterative single-channel loss with detach() and energy loss

# Observations
The model tends to create a background and foreground, i.e. a single
event to represent the common frequencies across the segment, and then
less frequency-rich events for the _differences_.  Aside from background
noise, I'd like every event to represent a real-world counterpart, and not
a fuzzy "background".  Perharps the hard gumbel softmax choice could help with this.

# Improvements

- add spectrograms to demo
- allow events to start _before_ the beginning of the sequence
- try without pure sine waves
- try with lower-resolution time-varying filter
- try nerf-like events with start time
- patch info loss
- try full run with gumbel softmax resonance selection



# Thesis

_Information_, or difference from the mean/expectation is precisely what makes
sounds rich, beautiful and interesting.  The learning objectives I'm using optimize
for precisely the _opposite_ of this;  they will always tend toward the mean/expectation
and remove the richness