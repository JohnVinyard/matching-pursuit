# this is multi-band matching pursuit

# from .e_2023_3_8 import BasicMatchingPursuit as Current
# from .e_2023_3_22 import MatchingPursuitGAN as Current


# from .e_2023_3_30 import ScalarPositioning as Current
# from .e_2023_3_30_B import BandFilteredImpulseResponse as Current


# from .e_2023_6_12 import PhaseInvariantFeatureInversion as Current

# from .e_2023_6_20 import SchedulingExperiment as Current
from .e_2023_6_27 import NoGridExperiment as Current

"""
Can I learn a better model (with dense latent variable)
with a sparse/matching pursuit loss?  Does the _loss_ need
to be sparse, and not the model?  Can a frame-based approach
work after all?
"""
# from .e_2023_7_13 import MatchingPursuitLoss as Current

# from .e_2023_7_18 import DenseToSparse as Current


# from .e_2023_7_20 import MatchingPursuitPlayground as Current

# from .e_2023_7_22 import SparseAutoencoder as Current

# TODO: Run overnight
# from .e_2023_7_25 import NeuralMatchingPursuit as Current



# from .e_2023_7_29 import HierarchicalScheduling as Current

"""
TODO Today

- model with momentum and damping that has an additional sparsity constraint
- another attempt at sparse bottleneck with different architecture

- multi-band matching pursuit (atoms "fused" across bands)
- matching pursuit with local contrast norm for atom selection
- matching pursuit in PIF space (3d atoms, requires inversion network)
- matching pursuit in "complex" frequency domain where we look at magnitude and deviance from expected phase advancement





- model that selects atom + resonance
- use a 3d self-organizing map to learn atoms so that pooling can happen across not just time
- hierarchical graph of schedulers
- sine optim experiment (coarse + fine)
"""