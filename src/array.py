from typing import Any
import numpy as np

from .core.math_engine import MathEngine
from .core.memory.base import MemoryLayout
from .core.memory.contiguous import ContiguousMemoryLayout
from .core.array_data.base import ArrayData
from .core.array_data.numpy_based import NumpyBasedArrayData

class NumpyArray:
    def __init__(self, data: Any = None, dtype=None):
        if isinstance(data, ArrayData):
            self._data = data
        else:
            np_array = np.asarray(data, dtype=dtype)
            self._data = NumpyBasedArrayData(np_array)

        self._math = MathEngine()

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def _unwrap(self, other: Any) -> ArrayData:
        if isinstance(other, NumpyArray):
            return other._data
        if isinstance(other, ArrayData):
            return other
        return NumpyBasedArrayData(np.asarray(other))  

    def __add__(self, other):
        other_data = self._unwrap(other)
        return NumpyArray(self._math.add(self._data, other_data))

    def __radd__(self, other):
        other_data = self._unwrap(other)
        return NumpyArray(self._math.add(other_data, self._data))

    def __sub__(self, other):
        other_data = self._unwrap(other)
        return NumpyArray(self._math.sub(self._data, other_data))

    def __rsub__(self, other):
        other_data = self._unwrap(other)
        return NumpyArray(self._math.sub(other_data, self._data))

    def __repr__(self):
        return f"NumpyArray(shape={self.shape}, dtype={self.dtype}, data={self._data.to_python_list()!r})"
