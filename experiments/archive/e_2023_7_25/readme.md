Observation:  While it has problems, matching pursuit seems to learn audio 
reproductions that sound better than _anything_ I've come up with that uses
a neural network and MSE loss, even when using a hand-engineered "perceptually-
motivated" transformation of the audio.  I _think_ this is because matching 
pursuit works in a highly-local way;  the full audio residual is never 
considered.

Matching pursuit:
    - sidesteps the problems caused by a frame-based audio representation
    - leads to a highly sparse and interpretible representation


That said, matching pursuit is _slow_, so any downstream task that relies
on the sparse representation either must just run slowly, or must pre-compute
the sparse representation over the entire dataset.

There are several steps I can take toward making a NN setting approximate
the matching pursuit approach.  

The first and most straightforward is as follows:
    1. create a NN that analyzes the signal and outputs a multi-channel signal with a gaussian window for each channel
    2. Each channel is subtracted from the signal **sequentially**.  The algorithm seeks to maximize the reduction 
        in norm only in the selected location, and not globally

Thinking more about it, there seem to be _two_ key characteristics:

- local vs. global losses
- the one-at-a-time matching pursuit update seems to encourage independence/orthogonality, 
    i.e., not every atom "leaps" in the same direction at once.