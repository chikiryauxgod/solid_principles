from abc import ABC, abstractmethod
from typing import Any, Tuple
from numpy import dtype as NumpyDType

Shape = Tuple[int, ...]


class ArrayData(ABC):
    @property
    @abstractmethod
    def shape(self) -> Shape: ...

    @property
    @abstractmethod
    def dtype(self) -> NumpyDType: ...

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def get_element(self, indices: Tuple[int, ...]) -> Any: ...

    @abstractmethod
    def set_element(self, indices: Tuple[int, ...], value: Any) -> None: ...

    @abstractmethod
    def to_python_list(self) -> list: ...

    @abstractmethod
    def copy(self) -> 'ArrayData': ...

    @abstractmethod
    def add(self, other: 'ArrayData') -> 'ArrayData':
        """self + other"""

    @abstractmethod
    def add_scalar(self, scalar: Any) -> 'ArrayData':
        """self + scalar"""

    @abstractmethod
    def subtract(self, other: 'ArrayData') -> 'ArrayData':
        """self - other"""

    @abstractmethod
    def subtract_scalar(self, scalar: Any) -> 'ArrayData':
        """self - scalar"""

    @abstractmethod
    def rsubtract_scalar(self, scalar: Any) -> 'ArrayData':
        """scalar - self"""
