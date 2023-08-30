from .erb import erb, scaled_erb

from .decompose import fft_frequency_decompose, fft_frequency_recompose
from .linear import LinearOutputStack
from .stft import geom_basis, stft, log_stft, stft_relative_phase, short_time_transform
from .psychoacoustic import PsychoacousticFeature
from .pos_encode import pos_encoded, ExpandUsingPosEncodings, LearnedPosEncodings, hard_pos_encoding
from .transformer import ForwardBlock, FourierMixer, Transformer
from .reverb import NeuralReverb, ReverbGenerator
# from .phase import AudioCodec, MelScale
from .pif import AuditoryImage
# from .metaformer import MetaFormer, PoolMixer, MetaFormerBlock
from .diffusion import DiffusionProcess
# from .atoms import AudioEvent
from .diffindex import diff_index
from .scattering import MoreCorrectScattering
from .normalization import UnitNorm, ExampleNorm, limit_norm, unit_norm, max_norm, MaxNorm
from .sparse import \
    VectorwiseSparsity, ElementwiseSparsity, AtomPlacement, \
    to_sparse_vectors_with_context, sparsify, sparsify_vectors, to_key_points, sparsify2
from .normal_pdf import pdf
from .waveguide import TransferFunctionSegmentGenerator
from .shape import Reshape
from .transfer import \
    TransferFunction, STFTTransferFunction, ImpulseGenerator, \
    PosEncodedImpulseGenerator, schedule_atoms, Position, fft_convolve, differentiable_fft_shift, scalar_position
from .fft import fft_shift, fft_convolve, simple_fft_convolve
from .physical import Window, harmonics, BlockwiseResonatorModel, scale_and_rotate
from .filter_bank import SynthesisBank
from .perceptual import PerceptualAudioModel
from .softmax import hard_softmax, sparse_softmax
from .matchingpursuit import \
    dictionary_learning_step, compare_conv, torch_conv, fft_convolve, \
    build_scatter_segments, flatten_atom_dict, sparse_feature_map, sparse_coding_loss, \
    SparseCodingLoss
from .pointcloud import encode_events, decode_events, extract_graph_edges, CanonicalOrdering
from .activation import unit_sine
from .floodfill import flood_fill_loss
from .hypernetwork import HyperNetworkLayer