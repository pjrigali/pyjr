"""
Handling errors.

Usage:
 ./utils/tools.py

Author:
 Peter Rigali - 2022-03-19
"""
from typing import Union, List, Optional
import numpy as np
import pandas as pd
import collections


# Cleaning Functions.
def _empty(data) -> bool:
    """Checks if data is empty"""
    if data.__len__() == 0:
        return True
    else:
        return False


def _to_list(data) -> list:
    """Converts list-adjacent objects to a list"""
    if isinstance(data, list):
        return data
    elif isinstance(data, pd.Series):
        return data.to_list()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (set, collections.abc.KeysView, collections.abc.ValuesView, tuple)):
        return list(data)
    elif isinstance(data, (int, float, str, object)):
        return [data]
    else:
        raise AttributeError('data needs to have a type of {np.ndarray, pd.Series, list, set, int, float, str, object}')


def _to_type(value: Union[float, int, str, object], dtype: str = 'float') -> Union[float, int, str, object]:
    """Converts value to a set type"""
    if isinstance(value, {'float': float, 'int': int, 'str': str, 'object': object}[dtype]):
        return value
    else:
        return {'float': float, 'int': int, 'str': str, 'object': object}[dtype](value)


def _check_type(data: list, dtype: str = 'float') -> tuple:
    """Checks type of values in a list"""
    return tuple([_to_type(value=val, dtype=dtype) for val in data])


def _check_na(value) -> bool:
    """
    Checks a value to see if Nan.

    :param value: Input value.
    :return: Returns True or False if the value is Nan.
    :rtype: bool.
    :note: *None*
    """
    if value == value and value is not None and value != np.inf and value != -np.inf:
        return False
    else:
        return True


def _remove_nan(data: list) -> list:
    """Remove Nan values from a list"""
    return [val for val in data if _check_na(val) is False]


def _round_to(data: Union[list, pd.Series, np.ndarray, float, int], val: float, remainder: bool = False) -> Union[list, float]:
    """
    Rounds a value or list.

    :param data: Value or list.
    :param val: Place to round to.
    :param remainder: Whether to use remainder. If using floats, this should be true.
    # :param val_type: Desired value type.
    # :type val_type: str.
    :return: Returns a value or list of values.
    :note: *None*
    """
    if isinstance(data, (float, int)):
        if remainder is True:
            return round(_to_type(value=data, dtype='float') * val) / val
        else:
            return round(_to_type(value=data, dtype='float') / val) * val
    elif isinstance(data, (list, pd.Series, np.ndarray)):
        data = (_to_type(value=i, dtype='float') for i in _to_list(data=data))
        if remainder is True:
            return [round(item * val) / val for item in data]
        else:
            return [round(item / val) * val for item in data]
    else:
        raise AttributeError('Value not one of the specified types.')


def _replacement_value(data: list, na_handling: str = 'median', std_value: int = 3, cap_zero: bool = True,
                       median_value: float = 0.023, ddof: int = 1) -> float:
    """
    Calculate desired replacement for Nan values.

    :param data: Input data.
    :type data: list.
    :param na_handling: Desired Nan value handling method. {zero, mu, std, median}
    :type na_handling: str.
    :param std_val: Desired Standard Deviation to use.
    :type std_val: int.
    :param cap_zero: Whether to cap the value at zero.
    :type cap_zero: bool.
    :param median_val: Desired percentile to use.
    :type median_val: float.
    :return: Replacement value.
    :note: If mean - 3 * std is less than 0, may confuse results.
    """
    if na_handling == 'zero':
        return 0.0
    elif na_handling == 'mu':
        return _mean(data=_remove_nan(data=data))
    elif na_handling == 'std':
        data = _remove_nan(data=data)
        val = _mean(data=data) - (_std(data=data, ddof=ddof) * std_value)
        if cap_zero:
            if val > 0:
                return val
            else:
                return 0.0
        else:
            return val
    elif na_handling == 'median':
        return _percentile(data=_remove_nan(data=data), q=median_value)
    elif na_handling == 'none':
        return None


def _replace_na(data: list, replacement_value: float = None) -> tuple:
    """Replace Nan values with replacement value"""
    if replacement_value is None:
        return _remove_nan(data=data)
    else:
        return tuple([val if _check_na(value=val) is False else replacement_value for val in data])


def _prep(data, dtype: str = 'float', na_handling: str = 'median', std_value: int = 3, median_value: float = 0.023,
          cap_zero: bool = True, ddof: int = 1):
    """Cleans data"""
    if _empty(data=data) is False:
        data = _to_list(data=data)
        na_value = _replacement_value(data=data, na_handling=na_handling, std_value=std_value,
                                      median_value=median_value, cap_zero=cap_zero, ddof=ddof)
        return _check_type(data=_replace_na(data=data, replacement_value=na_value), dtype=dtype)
    else:
        return None


# Non-cleaning related functions
def _add_constant(data: Union[list, tuple, np.ndarray]) -> np.ndarray:
    """Add a column of ones to a list, tuple or np.ndarray"""
    if isinstance(data, (tuple, list)):
        arr = np.ones((data.__len__(), 2))
    elif isinstance(data, np.ndarray):
        arr = np.ones((data.shape[0], 2))
    arr[:, 1] = data
    return arr


def _unique_values(data: list, count: False):
    """
    Finds unique values from a list.

    :param data: Input data.
    :type data: list.
    :return: Returns either a list or dict.
    :note: *None*
    """
    if count:
        unique = set(data)
        return {i: data.count(i) for i in unique}
    else:
        return tuple(set(data))


def _search_dic_values(dic: dict, item: Union[str, int, float]) -> Union[str, float, int]:
    """

    Searches a dict using the values.

    :param dic: Input data.
    :type dic: dict
    :param item: Search item.
    :type item: str, float or int
    :return: Key value connected to the value.
    :rtype: str, float or int
    :example: *None*
    :note: *None*

    """
    return list(dic.keys())[list(dic.values()).index(item)]


# Model_data
def _check_names(name, name_list) -> bool:
    name_dic = {name: True for name in name_list}
    if name not in name_dic:
        return True
    else:
        raise AttributeError("{} already included in names list".format(name))


def _check_len(len1, len2) -> bool:
    if len1 == len2:
        return True
    else:
        raise AttributeError("(len1: {} ,len2: {} ) Lengths are not the same.".format((len1, len2)))


def _add_column(arr1, arr2) -> np.ndarray:
    new_arr = np.ones((arr1.shape[0], arr1.shape[1] + 1))
    for i in range(arr1.shape[1]):
        new_arr[:, i] = arr1[:, i]
    new_arr[:, new_arr.shape[1] - 1] = arr2[:, 0]
    return new_arr


# FeaturePerformance
def _cent(x_lst: List[float], y_lst: List[float]) -> List[float]:
    """

    Calculate Centroid from x and y value(s).

    :param x_lst: A list of values.
    :type x_lst: List[float]
    :param y_lst: A list of values.
    :type y_lst: List[float]
    :returns: A list of x and y values representing the centriod of two lists.
    :rtype: List[float]
    :example: *None*
    :note: *None*

    """
    return [np.sum(x_lst) / len(x_lst), np.sum(y_lst) / len(y_lst)]


def _dis(cent1: List[float], cent2: List[float]) -> float:
    """

    Calculate Distance between two centroids.

    :param cent1: An x, y coordinate representing a centroid.
    :type cent1: List[float]
    :param cent2: An x, y coordinate representing a centroid.
    :type y_lst: List[float]
    :returns: A distance measurement.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    return round(np.sqrt((cent1[0] - cent2[0]) ** 2 + (cent1[1] - cent2[1]) ** 2), 4)

def stack(x_arr: np.ndarray, y_arr: np.ndarray, multi: Optional[bool] = False) -> np.ndarray:
    """

    Stacks x_arr and y_arr.

    :param x_arr: An array to stack.
    :type x_arr: np.ndarray
    :param y_arr: An array to stack.
    :type y_arr: np.ndarray
    :param mutli: If True, will stack based on multiple x_arr columns, default is False. *Optional*
    :type multi: bool
    :return: Array with a x column and a y column
    :rtype: np.ndarray
    :example: *None*
    :note: *None*

    """
    lst = []
    if multi:
        for i in range((x_arr.shape[1])):
            lst.append(np.vstack([x_arr[:, i].ravel(), y_arr[:, i].ravel()]).T)
        return np.array(lst)
    else:
        lst = np.vstack([x_arr.ravel(), y_arr.ravel()]).T
    return np.where(np.isnan(lst), 0, lst)
