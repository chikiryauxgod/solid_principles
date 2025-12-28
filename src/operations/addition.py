import numpy as np
from .base import MathOperation
from ..core.array_data.base import ArrayData
from ..core.array_data.numpy_based import NumpyBasedArrayData
from typing import Any

class Addition(MathOperation):
    @property
    def name(self) -> str:
        return "add"

    def execute(self, left: ArrayData, right: Any) -> ArrayData:
        if isinstance(right, ArrayData):
            return left.add(right)
        else:
            return left.add_scalar(right)