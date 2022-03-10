"""
Component functions.

Usage:
 ./utils/base.py

Author:
 Peter Rigali - 2022-03-10
"""
from typing import List, Optional, Union
import collections
import numpy as np
import pandas as pd
pd.set_option('use_inf_as_na', True)
class_dict = {'float': float, 'int': int, 'str': str, 'object': object}


def _to_list(data) -> list:
    """Converts list-adjacent objects to a list"""
    if isinstance(data, list):
        return data
    elif isinstance(data, pd.Series):
        return data.to_list()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (set, collections.abc.KeysView, collections.abc.ValuesView)):
        return list(data)
    elif isinstance(data, (int, float, str, object)):
        return [data]
    else:
        try:
            return list(data)
        except:
            raise AttributeError('data needs to have a type of {np.ndarray, pd.Series, list, set, int, float, str, object}')


def _check_list(data) -> list:
    """Checks if an object is a list"""
    return _to_list(data=data)


def _to_type(value: Union[float, int, str, object], value_type: str = 'float') -> Union[float, int, str, object]:
    """Converts value to a set type"""
    if isinstance(value, class_dict[value_type]):
        return value
    else:
        try:
            return class_dict[value_type](value)
        except:
            return value

def _check_type(data: list, value_type: str = 'float') -> list:
    """Checks type of values in a list"""
    return [_to_type(value=val, value_type=value_type) for val in data]


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
    datatype = type(data)

    if isinstance(data, (float, int)):
        if remainder is True:
            return round(_to_type(value=data, value_type='float') * val) / val
        else:
            return round(_to_type(value=data, value_type='float') / val) * val
    elif isinstance(data, (list, pd.Series, np.ndarray)):
        data = (_to_type(value=i, value_type='float') for i in _check_list(data=data))
        if remainder is True:
            return [round(item * val) / val for item in data]
        else:
            return [round(item / val) * val for item in data]
    else:
        raise AttributeError('Value not one of the specified types.')


def _unique_values(data: list, count: False) -> list:
    """
    Finds unique values from a list.

    :param data: Input data.
    :type data: list.
    :return: Returns either a list or dict.
    :note: *None*
    """
    if count:
        return {i: data.count(i) for i in set(data)}
    else:
        return _check_type(data=set(data), value_type='float')


# def _temp_unique_values(data: Union[list, np.ndarray, pd.Series],
#                         count: bool = False,
#                         order: bool = False,
#                         indexes: bool = False,
#                         na_handling: str = 'none',
#                         value_type: str = 'float',
#                         std_value: int = 3,
#                         median_value: float = 0.023,
#                         cap_zero: bool = True,
#                         ddof: int = 1):
#     """
#     Finds unique values from a list.
#
#     :param data: Input data.
#     :type data: list.
#     :param count: Whether to count the values.
#     :param order: Whether to sort() the values.
#     :param indexes: Whether to return the indexes.
#     :return: Returns either a list or dict.
#     :note: *None*
#     """
#     default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
#                'cap_zero': True, 'ddof': 1}
#     new_data = Args(args=(locals(), default)).data
#     if order:
#         temp_dic, temp_lst = {}, []
#         for item in new_data:
#             if item not in temp_dic:
#                 temp_dic[item] = True
#                 temp_lst.append(item)
#         return temp_lst
#     if count:
#         return {i: new_data.count(i) for i in set(new_data)}
#     if indexes:
#         temp_dic, ind_dic = {}, {}
#         for ind, item in enumerate(new_data):
#             if item in temp_dic:
#                 ind_dic[item].append(ind)
#             else:
#                 temp_dic[item] = True
#                 ind_dic[item] = [ind]
#         return ind_dic
#     return _to_list(data=set(new_data))


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


# Internal Math Functions
def _max(data: list) -> float:
    """
    Find the max value of a list.

    :param data: Input data.
    :type data: list.
    :return: Maximum value.
    :note: *None*
    """
    length = len(data)
    if length > 1:
        return _to_type(value=max(data), value_type='float')
    elif length == 0:
        return 0.0
    else:
        return _to_type(value=data, value_type='float')


def _min(data: list) -> float:
    """
    Find the min value of a list.

    :param data: Input data.
    :type data: list.
    :return: Minimum value.
    :note: *None*
    """
    length = len(data)
    if length > 1:
        return _to_type(value=min(data), value_type='float')
    elif length == 0:
        return 0.0
    else:
        return _to_type(value=data, value_type='float')


def _mean(data: list) -> float:
    """
    Find the mean value of a list.

    :param data: Input data.
    :type data: list.
    :return: Mean value.
    :rtype: float.
    :note: *None*
    """
    return sum(data) / len(data)


def _variance(data: list, ddof: int = 1) -> float:
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
    mu = _mean(data=data)
    return sum(((x - mu) ** 2 for x in data)) / (len(data) - ddof)


def _std(data: list, ddof: int = 1) -> float:
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
    return _variance(data=data, ddof=ddof) ** .5


def _sum(data: list) -> float:
    """
    Find the sum value of a list.

    :param data: Input data.
    :type data: list.
    :return: Sum value.
    :note: *None*
    """
    length = len(data)
    if length > 1:
        return _to_type(value=sum(data), value_type='float')
    elif length == 0:
        return 0.0
    else:
        return _to_type(value=data, value_type='float')


def _median(data: list) -> float:
    """
    Find the median value of a list.

    :param data: Input data.
    :type data: list.
    :return: Mean value.
    :rtype: float.
    :note: *None*
    """
    sorted_lst, lst_len = sorted(data), len(data)
    index = (lst_len - 1) // 2
    if lst_len % 2:
        return _to_type(value=sorted_lst[index], value_type='float')
    else:
        return _mean(data=[sorted_lst[index]] + [sorted_lst[index + 1]])


def _mode(data: list) -> float:
    """
    Find the mode value of a list.

    :param data: Input data.
    :type data: list.
    :return: Mode value.
    :rtype: float.
    :note: *None*
    """
    count_dic = _unique_values(data=data, count=True)
    count_dic_values = list(count_dic.values())
    dic_max = max(count_dic_values)
    lst = []
    for i in count_dic_values:
        val = _search_dic_values(dic=count_dic, item=dic_max)
        lst.append((val, i))
        # del count_dic[val]
        count_dic_values = list(count_dic.values())

    first_val, second_val = lst[0][0], lst[0][1]
    equal_lst = [i[0] for i in lst if second_val == i[1]]
    if len(equal_lst) == 1:
        return _to_type(value=first_val, value_type='float')
    elif len(equal_lst) % 2 == 0:
        return _mean(data=equal_lst)
    else:
        return _median(data=equal_lst)


def _skew(data: list, length: int) -> float:
    """
    Find the skew value of a list.

    :param data: Input data.
    :type data: list.
    :param length: Length of the input data.
    :type length: int.
    :return: Skew value.
    :rtype: float.
    :note: *None*
    """
    mu = _mean(data=data)
    stdn = _std(data=data, ddof=1) ** 3
    return (((_sum(data=[i - mu for i in data]) ** 3) / length) / stdn) * ((length * (length - 1)) ** .5) / (length - 2)


def _kurtosis(data: list, length: int) -> float:
    """
    Find the kurtosis value of a list.

    :param data: Input data.
    :type data: list.
    :param length: Length of the input data.
    :type length: int.
    :return: Kurtosis value.
    :rtype: float.
    :note: *None*
    """
    mu = _mean(data=data)
    stdn = _std(data=data, ddof=1) ** 4
    return (((_sum(data=[i - mu for i in data])**4) / length) / stdn) - 3


def _percentile(data: list, length: int, q: float, value_type: str = 'float') -> float:
    """
    Find the percentile value of a list.

    :param data: Input data.
    :type data: list.
    :param length: Length of the input data.
    :type length: int.
    :param q: Percentile percent.
    :type q: float.
    :param value_type: Desired type.
    :type value_type: str.
    :return: Percentile value.
    :note: *None*
    """
    if length == 0:
        return 0.0
    data = _round_to(data=[item * 1000.0 for item in data], val=1)
    ind = _round_to(data=length * q, val=1)
    data.sort()
    for item in data:
        if item >= data[ind]:
            return _to_type(value=item / 1000.0, value_type=value_type)


def _percentiles(data: list, length: int, q_lst: list = [0.159, 0.841], value_type: str = 'float'):
    """
    Calculate various percentiles for a list.

    :param data: Input data.
    :type data: list.
    :param length: Length of input data.
    :type length: int.
    :param q_lst: Desired percentile percents.
    :type q_lst: List of floats.
    :param value_type: Desired type.
    :type value_type: str.
    :return: A group of stats.
    :note: *None*
    """
    return (_percentile(data=data, length=length, q=q, value_type=value_type) for q in q_lst)


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
        data_new = _remove_nan(data=data)
        val = _mean(data=data_new) - (_std(data=data_new, ddof=ddof) * std_value)
        if cap_zero:
            if val > 0:
                return val
            else:
                return 0
        else:
            return val
    elif na_handling == 'median':
        return _percentile(data=data, length=len(data), q=median_value, value_type='float')
    elif na_handling == 'none':
        return None


def _replace_na(data: list, replacement_value: Optional[float] = None) -> list:
    """Replace Nan values with replacement value"""
    if replacement_value is None:
        return _remove_nan(data=data)
    else:
        return [val if _check_na(value=val) is False else replacement_value for val in data]


def _prep(data, value_type: str = 'float', na_handling: str = 'median', std_value: int = 3, median_value: float = 0.023,
          cap_zero: bool = True, ddof: int = 1):
    data_lst = _check_type(data=_check_list(data=data), value_type=value_type)
    na_value = _replacement_value(data=data_lst, na_handling=na_handling, std_value=std_value,
                                  median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    return _replace_na(data=data_lst, replacement_value=na_value)

