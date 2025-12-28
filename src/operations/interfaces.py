from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray, ArrayLike

class MathOperation(ABC):
    """Basic interface to math operation"""
    
    @abstractmethod
    def execute(self, left: NDArray, right: ArrayLike) -> NDArray:
        pass
