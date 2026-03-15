from typing import NewType, TypeVar


BatchDim = NewType('BatchDim', int)
IdentityDim = NewType('IdentityDim', int)
TimeDim = NewType('TimeDim', int)
ChannelsDim = NewType('ChannelsDim', int)
EventsDim = NewType('EventsDim', int)
SizeDim = NewType('SizeDim', int)

E = TypeVar('E', EventsDim, IdentityDim)
B = TypeVar('B', BatchDim, IdentityDim)
T = TypeVar('T', TimeDim, IdentityDim)
C = TypeVar('C', ChannelsDim, IdentityDim)
S = TypeVar('S', SizeDim, IdentityDim)
I = TypeVar('I', IdentityDim, TimeDim, BatchDim, ChannelsDim, SizeDim, EventsDim)





