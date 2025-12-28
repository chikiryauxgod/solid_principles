import pytest
import numpy as np

from src.array import NumpyArray


def test_add_two_arrays():
    a = NumpyArray([1, 2, 3])
    b = NumpyArray([10, 20, 30])
    
    result = a + b
    
    assert isinstance(result, NumpyArray)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result._data.to_python_list(), [11, 22, 33])


def test_add_scalar():
    a = NumpyArray([5, 6, 7])
    result = a + 100
    
    np.testing.assert_array_equal(result._data.to_python_list(), [105, 106, 107])


def test_radd_scalar():
    result = 100 + NumpyArray([1, 2, 3])
    
    np.testing.assert_array_equal(result._data.to_python_list(), [101, 102, 103])


def test_subtract_arrays():
    a = NumpyArray([10, 20, 30])
    b = NumpyArray([1, 2, 3])
    
    result = a - b
    np.testing.assert_array_equal(result._data.to_python_list(), [9, 18, 27])


def test_rsubtract_scalar():
    result = 100 - NumpyArray([30, 20, 10])
    
    np.testing.assert_array_equal(result._data.to_python_list(), [70, 80, 90])


def test_shape_ndim_size():
    a = NumpyArray([[1, 2], [3, 4], [5, 6]])
    
    assert a.shape == (3, 2)
    assert a.ndim == 2
    assert a.size == 6