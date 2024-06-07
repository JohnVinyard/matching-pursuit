from typing import Any, BinaryIO, Dict, Union
from conjure.serialize import Serializer, Deserializer
from conjure import Conjure, conjure, ParamsHash, FunctionContentIdentifier, SupportedContentType
from conjure.storage import Collection
from torch.nn import Module
from io import BytesIO
import torch
from torch import Tensor

StateDict = Dict[str, Union[Module, Tensor]]

class TorchModuleSerializer(Serializer):
    
    def write(self, content: StateDict, sink: BinaryIO) -> None:
        torch.save(content, sink)
    
    def to_bytes(self, content: StateDict) -> bytes:
        bio = BytesIO()
        torch.save(content, bio)
        bio.seek(0)
        return bio.read()


class TorchModuleDeserializer(Deserializer):
    
    def read(self, sink: BinaryIO) -> StateDict:
        return torch.load(sink)
    
    def from_bytes(self, encoded: bytes) -> StateDict:
        bio = BytesIO()
        bio.write(encoded)
        bio.seek(0)
        return torch.load(bio)


def torch_conjure(storage: Collection):
    return conjure(
        content_type='application/octet-stream',
        storage=storage,
        func_identifier=FunctionContentIdentifier(),
        param_identifier=ParamsHash(),
        serializer=TorchModuleSerializer(),
        deserializer=TorchModuleDeserializer(),
    )