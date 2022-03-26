"""
Handling errors.

Usage:
 ./utils/tools.py

Author:
 Peter Rigali - 2022-03-19
"""
from typing import Union, List, Optional
import numpy as np
from collections.abc import KeysView, ValuesView
from pandas import Series, DataFrame


# Repeat functions
def _mean_(data: Union[list, tuple]) -> float:
    """
    Find the mean value of a list.

    :param data: Input data.
    :type data: list.
    :return: Mean value.
    :rtype: float.
    :note: *None*
    """
    return sum(data) / data.__len__()


def _variance_(data: Union[list, tuple], ddof: int = 1) -> float:
    """
    Find the variance value of a list.

    :param data: Input data.
    :type data: list.
    :param ddof: Desired Degrees of Freedom.
    :type ddof: int
    :return: Variance value.
    :rtype: float.
    :note: *None*
    """
    mu = _mean_(data=data)
    return sum(((x - mu) ** 2 for x in data)) / (data.__len__() - ddof)


def _std_(data: list, ddof: int = 1) -> float:
    """
    Find the Standard Deviation value of a list.

    :param data: Input data.
    :type data: list.
    :param ddof: Desired Degrees of Freedom.
    :type ddof: int
    :return: Standard Deviation value.
    :rtype: float.
    :note: *None*
    """
    return _variance_(data=data, ddof=ddof) ** .5


def _percentile_(data: Union[list, tuple], q: float) -> float:
    """
    Find the percentile value of a list.

    :param data: Input data.
    :type data: list.
    :param q: Percentile percent.
    :type q: float.
    :return: Percentile value.
    :note: *None*
    """
    data = _round_to(data=[item * 1000.0 for item in data], val=1)
    ind = _round_to(data=data.__len__() * q, val=1)
    data.sort()
    for item in data:
        if item >= data[ind]:
            return item / 1000.0


# Cleaning Functions.
def _empty(data) -> bool:
    """Checks if data is empty"""
    if data.__len__() == 0:
        return True
    else:
        return False


def _to_metatype(data, dtype: str = 'list') -> Union[list, tuple]:
    """Converts list-adjacent objects to a list or tuple"""
    if dtype not in {'list': True, 'tuple': True}:
        raise AttributeError("dtype input must be either {list, tuple}.")
    if isinstance(data, {'list': list, 'tuple': tuple}[dtype]):
        return data
    elif isinstance(data, Series):
        if dtype == 'list':
            return data.to_list()
        else:
            return tuple(data.to_list())
    elif isinstance(data, (np.ndarray, DataFrame)):
        if dtype == 'list':
            return data.tolist()
        else:
            return tuple(data.tolist())
    elif isinstance(data, (set, KeysView, ValuesView)):
        return {'list': list, 'tuple': tuple}[dtype](data)
    elif isinstance(data, (int, float, str, object, np.int_, np.float_, np.str, np.object)):
        if dtype == 'list':
            return [data]
        else:
            return tuple(data)
    else:
        raise AttributeError('Input data needs to have a type of {np.ndarray, pd.Series, list, set, int, float, str, object}')


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


def _remove_nan(data: Union[list, tuple]) -> list:
    """Remove Nan values from a list"""
    return [val for val in data if _check_na(val) is False]


def _round_to(data: Union[list, Series, np.ndarray, float, int], val: float,
              remainder: bool = False) -> Union[list, float]:
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
    elif isinstance(data, (list, Series, np.ndarray)):
        data = (_to_type(value=i, dtype='float') for i in _to_metatype(data=data))
        if remainder is True:
            return [round(item * val) / val for item in data]
        else:
            return [round(item / val) * val for item in data]
    else:
        raise AttributeError('Value not one of the specified types.')


def _replacement_value(data: list, na_handling: str = 'median', std_value: int = 3, cap_zero: bool = True,
                       median_value: float = 0.023, ddof: int = 1) -> Union[float, None]:
    """
    Calculate desired replacement for Nan values.

    :param data: Input data.
    :type data: list.
    :param na_handling: Desired Nan value handling method. {zero, mu, std, median}
    :type na_handling: str.
    :param std_value: Desired Standard Deviation to use.
    :type std_value: int.
    :param cap_zero: Whether to cap the value at zero.
    :type cap_zero: bool.
    :param median_value: Desired percentile to use.
    :type median_value: float.
    :return: Replacement value.
    :note: If mean - 3 * std is less than 0, may confuse results.
    """
    if na_handling == 'zero':
        return 0.0
    elif na_handling == 'mu':
        return _mean_(data=_remove_nan(data=data))
    elif na_handling == 'std':
        data = _remove_nan(data=data)
        val = _mean_(data=data) - (_std_(data=data, ddof=ddof) * std_value)
        if cap_zero:
            if val > 0:
                return val
            else:
                return 0.0
        else:
            return val
    elif na_handling == 'median':
        return _percentile_(data=_remove_nan(data=data), q=median_value)
    elif na_handling == 'none':
        return None


def _replace_na(data: Union[list, tuple], replacement_value: float = None) -> Union[list, tuple]:
    """Replace Nan values with replacement value"""
    if replacement_value is None:
        return _remove_nan(data=data)
    else:
        return tuple([val if _check_na(value=val) is False else replacement_value for val in data])


# def _prep(data, dtype: str = 'float', na_handling: str = 'median', std_value: int = 3, median_value: float = 0.023,
#           cap_zero: bool = True, ddof: int = 1):
#     """Cleans data"""
#     if _empty(data=data) is False:
#         data = _to_metatype(data=data)
#         na_value = _replacement_value(data=data, na_handling=na_handling, std_value=std_value,
#                                       median_value=median_value, cap_zero=cap_zero, ddof=ddof)
#         return _check_type(data=_replace_na(data=data, replacement_value=na_value), dtype=dtype)
#     else:
#         return None


def _prep(data, meta_type: str = 'tuple', dtype: str = 'float', na_handling: str = 'zero', std_value: int = 3,
          median_value: float = 0.023, cap_zero: bool = True, ddof: int = 1):
    """Clean data"""
    # Check Empty
    if _empty(data=data):
        raise AttributeError("Data entered is empty.")
    # Convert to list
    data = _to_metatype(data=data, dtype='list')
    # Check type, check na, replace na
    na = None
    for ind, val in enumerate(data):
        if _check_na(value=val):
            if na is None:
                na = _replacement_value(data=data, na_handling=na_handling)
            val = na
        data[ind] = _to_type(value=val, dtype=dtype)
    # Convert to metadata
    data = _to_metatype(data=data, dtype=meta_type)
    return data


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
        raise AttributeError("(len1: {} ,len2: {} ) Lengths are not the same.".format(len1, len2))


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
    :type cent2: List[float]
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
    :param multi: If True, will stack based on multiple x_arr columns, default is False. *Optional*
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
