"""
Cleaning functions.

Usage:
 ./utils/_tools/clean.py

Author:
 Peter Rigali - 2022-03-30
"""
from typing import Union
from collections.abc import KeysView, ValuesView
from pandas import Series, Index
import numpy as np


# Repeat functions
def _mean_(d: Union[list, tuple]) -> float:
    """
    Find the mean value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Mean value.
    :rtype: float.
    :note: *None*
    """
    return sum(d) / d.__len__()


def _variance_(d: Union[list, tuple], ddof: int = 1) -> float:
    """
    Find the variance value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :param ddof: Desired Degrees of Freedom.
    :type ddof: int
    :return: Variance value.
    :rtype: float.
    :note: *None*
    """
    mu = _mean_(d=d)
    return sum(((x - mu) ** 2 for x in d)) / (d.__len__() - ddof)


def _std_(d: list, ddof: int = 1) -> float:
    """
    Find the Standard Deviation value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :param ddof: Desired Degrees of Freedom.
    :type ddof: int
    :return: Standard Deviation value.
    :rtype: float.
    :note: *None*
    """
    return _variance_(d=d, ddof=ddof) ** .5


def _percentile_(d: Union[list, tuple], q: float) -> float:
    """
    Find the percentile value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :param q: Percentile percent.
    :type q: float.
    :return: Percentile value.
    :note: *None*
    """
    d = _round(d=[item * 1000.0 for item in d], v=1)
    ind = _round(d=d.__len__() * q, v=1)
    d.sort()
    for item in d:
        if item >= d[ind]:
            return item / 1000.0


# Misc
def _round(d: Union[list, Series, np.ndarray, float, int], v: float,
           r: bool = False) -> Union[list, float]:
    """
    Rounds a value or list.

    :param d: Value or list.
    :param v: Place to round to.
    :param r: Whether to use remainder. If using floats, this should be true.
    # :param val_type: Desired value type.
    # :type val_type: str.
    :return: Returns a value or list of values.
    :note: *None*
    """
    if isinstance(d, (float, int)):
        if r is True:
            return round(_type(v=d, dtype='float') * v) / v
        else:
            return round(_type(v=d, dtype='float') / v) * v
    elif isinstance(d, (list, Series, np.ndarray)):
        d = (_type(v=i, dtype='float') for i in _mtype(d=d))
        if r is True:
            return [round(item * v) / v for item in d]
        else:
            return [round(item / v) * v for item in d]
    else:
        raise AttributeError('Value not one of the specified types.')


# Cleaning Functions.
def _empty(d) -> bool:
    """Checks if data is empty"""
    if d.__len__() == 0:
        return True
    else:
        return False


def _mtype(d, dtype: str = 'list') -> Union[list, tuple]:
    """Converts list-adjacent objects to a list or tuple"""
    if dtype not in {'list': True, 'tuple': True}:
        raise AttributeError("dtype input must be either {list, tuple}.")
    if isinstance(d, {'list': list, 'tuple': tuple}[dtype]):
        return d
    elif isinstance(d, Series):
        if dtype == 'list':
            return d.to_list()
        else:
            return tuple(d.to_list())
    elif isinstance(d, np.ndarray):
        if dtype == 'list':
            return d.tolist()
        else:
            return tuple(d.tolist())
    elif isinstance(d, (set, KeysView, ValuesView, Index)):
        return {'list': list, 'tuple': tuple}[dtype](d)
    elif isinstance(d, (int, float, str, object, np.int_, np.float_, np.str, np.object)):
        if dtype == 'list':
            return [d]
        else:
            return tuple(d)
    else:
        raise AttributeError('Input data needs to have a type of {np.ndarray, pd.Series, list, set, int, float, str, object}')


def _type(v: Union[float, int, str, object], dtype: str = 'float') -> Union[float, int, str, object]:
    """Converts value to a set type"""
    if isinstance(v, {'float': float, 'int': int, 'str': str, 'object': object}[dtype]):
        return v
    else:
        return {'float': float, 'int': int, 'str': str, 'object': object}[dtype](v)


def _check_type(d: Union[list, tuple], dtype: str = 'float') -> tuple:
    """Checks type of values in a list"""
    return tuple([_type(v=val, dtype=dtype) for val in d])


def _nan(v) -> bool:
    """
    Checks a value to see if Nan.

    :param v: Input value.
    :return: Returns True or False if the value is Nan.
    :rtype: bool.
    :note: *None*
    """
    if v == v and v is not None and v != np.inf and v != -np.inf:
        return False
    else:
        return True


def _rnan(d: Union[list, tuple]) -> list:
    """Remove Nan values from a list"""
    return [val for val in d if _nan(val) is False]


def _rval(d: list, na: str = 'median', std_value: int = 3, cap_zero: bool = True,
          median_value: float = 0.023, ddof: int = 1) -> Union[float, None]:
    """
    Calculate desired replacement for Nan values.

    :param d: Input data.
    :type d: list.
    :param na: Desired Nan value handling method. {zero, mu, std, median}
    :type na: str.
    :param std_value: Desired Standard Deviation to use.
    :type std_value: int.
    :param cap_zero: Whether to cap the value at zero.
    :type cap_zero: bool.
    :param median_value: Desired percentile to use.
    :type median_value: float.
    :return: Replacement value.
    :note: If mean - 3 * std is less than 0, may confuse results.
    """
    if na == 'zero':
        return 0.0
    elif na == 'mu':
        return _mean_(d=_rnan(d=d))
    elif na == 'std':
        d = _rnan(d=d)
        val = _mean_(d=d) - (_std_(d=d, ddof=ddof) * std_value)
        if cap_zero:
            if val > 0:
                return val
            else:
                return 0.0
        else:
            return val
    elif na == 'median':
        return _percentile_(d=_rnan(d=d), q=median_value)
    elif na == 'none':
        return None


def _replace_na(d: Union[list, tuple], rval: float = None) -> Union[list, tuple]:
    """Replace Nan values with replacement value"""
    if rval is None:
        return _rnan(d=d)
    else:
        return tuple([val if _nan(v=val) is False else rval for val in d])


def _prep(d, mtype: str = 'tuple', dtype: str = 'float', na: str = 'zero', std_value: int = 3,
          median_value: float = 0.023, cap_zero: bool = True, ddof: int = 1):
    """Clean data"""
    # Check Empty
    if _empty(d=d):
        raise AttributeError("Data entered is empty.")
    # Convert to list
    d = _mtype(d=d, dtype='list')
    # Check type, check na, replace na
    _na = None
    for ind, val in enumerate(d):
        if _nan(v=val):
            if _na is None:
                _na = _rval(d=d, na=na)
            val = _na
        d[ind] = _type(v=val, dtype=dtype)
    # Convert to metadata
    d = _mtype(d=d, dtype=mtype)
    return d
