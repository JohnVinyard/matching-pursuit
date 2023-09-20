# from .e_2023_3_22 import MatchingPursuitGAN as Current
# from .e_2023_3_30_B import BandFilteredImpulseResponse as Current

# from .e_2023_6_12 import PhaseInvariantFeatureInversion as Current

# from .e_2023_6_27 import NoGridExperiment as Current
# from .e_2023_7_20 import MatchingPursuitPlayground as Current

# from .e_2023_7_25 import NeuralMatchingPursuit as Current

# from .e_2023_8_2 import FrameBasedModel as Current


# from .e_2023_3_8 import BasicMatchingPursuit as Current

# from .e_2023_3_30 import ScalarPositioning as Current


# from .e_2023_6_20 import SchedulingExperiment as Current


# from .e_2023_7_13 import MatchingPursuitLoss as Current

# from .e_2023_7_18 import DenseToSparse as Current


# from .e_2023_7_22 import SparseAutoencoder as Current


# HERE IS THE OTHER SCHEDULING THING
# from .e_2023_7_29 import HierarchicalScheduling as Current

# from .e_2023_8_1 import NerfContinuation as Current


# from .e_2023_8_4 import SparseAdversarialLoss as Current

# from .e_2023_8_6 import AdversarialSparseScheduler as Current


# from .e_2023_8_7 import TryingStuffOut as Current


# from .e_2023_8_8 import SparseStreamingCodec as Current

# from .e_2023_8_13 import F0Optim as Current

# from .e_2023_8_25 import HyperNerf as Current

# from .e_2023_8_31 import SparsityPenalty as Current

# from .e_2023_9_4 import SparseResonance as Current

# from .e_2023_9_8 import SparseResonance2 as Current

# from .e_2023_9_17 import SparseV3 as Current

# from .e_2023_9_18 import SparseV4 as Current

from .e_2023_9_19 import MatchingPursuitV3 as Current

"""
Features of Next Experiment
============================
- analysis happens at frame and not sample level; grinding sounds are a result of sample-level analysis, I think
- two options:
    - sparsification happens at frame level and then upsample with traditional transposed conv
    - upsampling happens, then sparsification, then fft convolution
- add in fft_shit, but **relative to frame location**
- lateral competition when sparsifying

## Model
- pif feature (Q: would "more-correct" scattering be better?)
- analysis stack
- upsample to sample-rate (either nearest-neighbor, or "local NERF" with overlap-add)
- sparsification, ideally with lateral competition (how do I handle feature similarity?) 
    - (https://hal.science/hal-00727563/document#page=13)
    - could I just "sort" (random projection to N-D) kernels after every batch? 
- generate dictionaries and convolve with events


## Loss Ideas
- HPSS loss
- sort active channels and optimize the contribution of each, going from largest to smallest?


Findings
==============
- sparse generation helps immensely with more natural-sounding audio
- fft_shift scheduling is possible when using proper normalization and not limiting values with a non-linearity


TODO

transposed conv that grows from start of signal rather than the middle

- model with momentum and damping that has an additional sparsity constraint
    - in other words, minimize energy

- another attempt at sparse bottleneck with different architecture

- multi-band matching pursuit (atoms "fused" across bands)
- matching pursuit with local contrast norm for atom selection
- matching pursuit in PIF space (3d atoms, requires inversion network)
- matching pursuit in "complex" frequency domain where we look at magnitude and deviance from expected phase advancement



There's something special/important about the one-by-one atom update of matching pursuit.
When moving away from a grid-based to a scheduler based representation, the space of possible 
solutions explodes.  I think this is one reason the loss landspace is so spiky and unstable

- model that selects atom + resonance
- use a 3d self-organizing map to learn atoms so that pooling can happen across not just time
- hierarchical graph of schedulers
- sine optim experiment (coarse + fine)
"""
