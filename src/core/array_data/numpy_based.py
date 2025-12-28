import numpy as np
from .base import ArrayData, Shape
from typing import Any, Tuple

class NumpyBasedArrayData(ArrayData):
    def __init__(self, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise TypeError("Expected numpy.ndarray")
        self._array = array

    @property
    def shape(self) -> Shape:
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def size(self) -> int:
        return self._array.size

    def get_element(self, indices: Tuple[int, ...]) -> Any:
        return self._array[indices]

    def set_element(self, indices: Tuple[int, ...], value: Any) -> None:
        self._array[indices] = value

    def to_python_list(self) -> list:
        return self._array.tolist()

    def copy(self) -> 'NumpyBasedArrayData':
        return NumpyBasedArrayData(self._array.copy())

    def reshape(self, new_shape: Shape) -> 'NumpyBasedArrayData':
        return NumpyBasedArrayData(self._array.reshape(new_shape))
    
    def add(self, other: 'ArrayData') -> 'ArrayData':
        if not isinstance(other, NumpyBasedArrayData):
            raise TypeError("Supports only NumpyBasedArrayData")
        return NumpyBasedArrayData(self._array + other._array)

    def add_scalar(self, scalar: Any) -> 'ArrayData':
        return NumpyBasedArrayData(self._array + scalar)

    def subtract(self, other: 'ArrayData') -> 'ArrayData':
        if not isinstance(other, NumpyBasedArrayData):
            raise TypeError("Supports only NumpyBasedArrayData")
        return NumpyBasedArrayData(self._array - other._array)

    def subtract_scalar(self, scalar: Any) -> 'ArrayData':
        return NumpyBasedArrayData(self._array - scalar)

    def rsubtract_scalar(self, scalar: Any) -> 'ArrayData':
        return NumpyBasedArrayData(scalar - self._array)