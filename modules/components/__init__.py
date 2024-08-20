"""
This components module should become the new, public interface for the 
parent modules
"""

from .auditory import STFTTransform
from .linear import LinearOutputStack
from .upsample import ConvUpsample
from .signal import make_waves