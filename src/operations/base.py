from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from numpy.typing import NDArray, ArrayLike


class MathOperation(ABC):
    """
    Abstract class to all math operations
    """

    @abstractmethod
    def execute(self, left: NDArray[Any], right: Any) -> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __str__(self) -> str:
        return self.name