import numpy as np
from typing import Any

from .base import MathOperation
from ..core.array_data.base import ArrayData
from ..core.array_data.numpy_based import NumpyBasedArrayData


class Subtraction(MathOperation):
    @property
    def name(self) -> str:
        return "sub"

    def execute(self, left: ArrayData, right: Any) -> ArrayData:
        if isinstance(right, ArrayData):
            return left.subtract(right)
        else:
            return left.subtract_scalar(right)