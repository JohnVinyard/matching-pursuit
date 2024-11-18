from abc import ABC, abstractmethod
from typing import Dict, Tuple

ShapeSpec = Dict[str, Tuple[int]]

class EventGenerator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def shape_spec(self) -> ShapeSpec:
        raise NotImplementedError()

    @abstractmethod
    def random_sequence(self):
        raise NotImplementedError()