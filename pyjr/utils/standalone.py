"""
Stand-alone functions.

Usage:
 ./utils/standalone.py

Author:
 Peter Rigali - 2022-03-10
"""
import pandas as pd
import numpy as np
from typing import Union
from pyjr.utils.base import _prep, _unique_values, _to_list, _search_dic_values, _to_type, _max, _check_type, _round_to
from pyjr.utils.base import _min
from pyjr.utils.args import native_mean_args, native_median_args, native_variance_args, native_std_args, native_sum_args
from pyjr.utils.args import _prep_args
from pyjr.utils.input import Args

# def flatten(data: list, type_used: str = 'str') -> list:
#     """
#
#     Flattens a list and checks the list.
#
#     :param data: Input data.
#     :type data: list
#     :param type_used: Type to search for, default is "str". *Optional*
#     :type type_used: str
#     :param type_used: Either {str, int, or float}
#     :type type_used: str
#     :return: Returns a flattened list.
#     :rtype: list
#     :example: *None*
#     :note: Will work when lists are mixed with non-list items.
#
#     """
#     new_type = {'str': [str], 'int': [int], 'float': [float], 'class objects': class_object_lst}[type_used]
#     lst = [item1 for item1 in data if type(item1) in new_type or item1 is None]
#     missed = [item1 for item1 in data if type(item1) not in new_type and item1 is not None]
#     temp_lst = [item2 for item1 in missed for item2 in item1]
#     return lst + temp_lst


def unique_values(data: Union[list, np.ndarray, pd.Series],
                  count: bool = False,
                  order: bool = False,
                  indexes: bool = False,
                  na_handling: str = 'none',
                  value_type: str = 'float',
                  std_value: int = 3,
                  median_value: float = 0.023,
                  cap_zero: bool = True,
                  ddof: int = 1):
    """
    Finds unique values from a list.

    :param data: Input data.
    :type data: list.
    :param count: Whether to count the values.
    :param order: Whether to sort() the values.
    :param indexes: Whether to return the indexes.
    :return: Returns either a list or dict.
    :note: *None*
    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default)).data
    if order:
        temp_dic, temp_lst = {}, []
        for item in new_data:
            if item not in temp_dic:
                temp_dic[item] = True
                temp_lst.append(item)
        return temp_lst
    if count:
        return {i: new_data.count(i) for i in set(new_data)}
    if indexes:
        temp_dic, ind_dic = {}, {}
        for ind, item in enumerate(new_data):
            if item in temp_dic:
                ind_dic[item].append(ind)
            else:
                temp_dic[item] = True
                ind_dic[item] = [ind]
        return ind_dic
    return _to_list(data=set(new_data))


def native_mode(data: Union[list, np.ndarray, pd.Series],
                na_handling: str = 'none',
                value_type: str = 'float',
                std_value: int = 3,
                median_value: float = 0.023,
                cap_zero: bool = True,
                ddof: int = 1) -> Union[float, int]:
    """

    Calculate Mode of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the Mode.
    :rtype: float
    :example: *None*
    :note: *None*
    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    count_dic = unique_values(data=new_data.data, count=True)
    count_dic_values = _to_list(data=count_dic.values())
    lst, dic_max = [], _max(count_dic_values)
    for i in count_dic_values:
        lst.append((_search_dic_values(dic=count_dic, item=dic_max), i))
        count_dic_values = _to_list(data=count_dic.values())
        continue
        # val = _search_dic_values(dic=count_dic, item=dic_max)
        # lst.append((val, i))
        # # del count_dic[val]
        # count_dic_values = _to_list(data=count_dic.values())

    first_val, second_val = lst[0][0], lst[0][1]
    equal_lst = [i[0] for i in lst if second_val == i[1]]
    new_args = (equal_lst, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    lst_len = len(equal_lst)
    if lst_len == 1:
        return float(first_val)
    elif lst_len % 2 == 0:
        return native_mean_args(args=new_args)
    else:
        return native_median(data=equal_lst, value_type=value_type)


def native_median(data: Union[list, np.ndarray, pd.Series],
                na_handling: str = 'none',
                value_type: str = 'float',
                std_value: int = 3,
                median_value: float = 0.023,
                cap_zero: bool = True,
                ddof: int = 1) -> Union[float, int]:
    """

    Calculate Median of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the Median.
    :rtype: float
    :example: *None*
    :note: If multiple values have the same count, will return the mean.
        Median is used if there is an odd number of same count values.

    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    sorted_lst = sorted(new_data.data)
    index = (new_data.len - 1) // 2
    if new_data.len % 2:
        return _to_type(value=sorted_lst[index], value_type=value_type)
    else:
        temp_data = [sorted_lst[index]] + [sorted_lst[index + 1]]
        new_args = (temp_data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
        return native_mean_args(args=args)


def native_mean(data: Union[list, np.ndarray, pd.Series],
                na_handling: str = 'none',
                value_type: str = 'float',
                std_value: int = 3,
                median_value: float = 0.023,
                cap_zero: bool = True,
                ddof: int = 1) -> Union[float, int]:
    """

    Calculate Mean of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the mean.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    if new_data.len != 0:
        return _to_type(value=new_data.sum / new_data.len, value_type=value_type)
    else:
        return _to_type(value=0.0, value_type=value_type)


def native_variance(data: Union[list, np.ndarray, pd.Series],
                    na_handling: str = 'none',
                    value_type: str = 'float',
                    std_value: int = 3,
                    median_value: float = 0.023,
                    cap_zero: bool = True,
                    ddof: int = 1) -> Union[float, int]:
    """

    Calculate Variance of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param ddof: Set the degrees of freedom, default is 1. *Optional*
    :type ddof: int
    :return: Returns the Variance.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    new_args = (new_data.data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    mu = native_mean_args(args=new_args)
    return _to_type(value=sum((x - mu) ** 2 for x in new_data) / (new_data.len - ddof), value_type=value_type)


def native_std(data: Union[list, np.ndarray, pd.Series],
               na_handling: str = 'none',
               value_type: str = 'float',
               std_value: int = 3,
               median_value: float = 0.023,
               cap_zero: bool = True,
               ddof: int = 1) -> Union[float, int]:
    """

    Calculate Standard Deviation of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param ddof: Set the degrees of freedom, default is 1. *Optional*
    :type ddof: int
    :return: Returns the Standard Deviation.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default)).data
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (new_data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    return native_variance_args(args=args) ** .5


def native_sum(data: Union[list, np.ndarray, pd.Series],
               na_handling: str = 'none',
               value_type: str = 'float',
               std_value: int = 3,
               median_value: float = 0.023,
               cap_zero: bool = True,
               ddof: int = 1) -> Union[float, int]:
    """

    Calculate Sum of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the Sum.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # data = _prep_args(args=args)
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    if new_data.len > 1:
        return new_data.sum
    elif new_data.len == 0:
        return 0.0
    else:
        return new_data


def native_max(data: Union[list, np.ndarray, pd.Series],
               na_handling: str = 'none',
               value_type: str = 'float',
               std_value: int = 3,
               median_value: float = 0.023,
               cap_zero: bool = True,
               ddof: int = 1) -> Union[float, int]:
    """

    Calculate Max of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the max value.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    if new_data.len > 1:
        return new_data.max
    elif new_data.len == 0:
        return 0.0
    else:
        return data


def native_min(data: Union[list, np.ndarray, pd.Series],
               na_handling: str = 'none',
               value_type: str = 'float',
               std_value: int = 3,
               median_value: float = 0.023,
               cap_zero: bool = True,
               ddof: int = 1) -> Union[float, int]:
    """

    Calculate Min of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the min value.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    if new_data.len > 1:
        return new_data.min
    elif new_data.len == 0:
        return 0.0
    else:
        return data


def native_skew(data: Union[list, np.ndarray, pd.Series],
               na_handling: str = 'none',
               value_type: str = 'float',
               std_value: int = 3,
               median_value: float = 0.023,
               cap_zero: bool = True,
               ddof: int = 1) -> Union[float, int]:
    """

    Calculate Skew of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the skew value.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    temp_args = (new_data.data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    mu, stdn = native_mean_args(args=temp_args), native_std_args(args=temp_args) ** 3
    new_args = ([i - mu for i in new_data.data], value_type, na_handling, std_value, median_value, cap_zero, ddof)
    return (((native_sum_args(args=new_args) ** 3) / new_data.len) / stdn) * ((new_data.len * (new_data.len - 1)) ** .5) / (new_data.len - 2)


def native_kurtosis(data: Union[list, np.ndarray, pd.Series],
                    na_handling: str = 'none',
                    value_type: str = 'float',
                    std_value: int = 3,
                    median_value: float = 0.023,
                    cap_zero: bool = True,
                    ddof: int = 1) -> Union[float, int]:
    """

    Calculate Kurtosis of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Returns the kurtosis value.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # data = _prep_args(args=args)
    temp_args = (new_data.data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    mu, stdn = native_mean_args(args=temp_args), native_std_args(args=args) ** 4
    new_args = ([i - mu for i in new_data.data], value_type, na_handling, std_value, median_value, cap_zero, ddof)
    return (((native_sum_args(args=new_args) ** 4) / new_data.len) / stdn) - 3


def native_percentile(data: Union[list, np.ndarray, pd.Series],
                      q: float,
                      na_handling: str = 'none',
                      value_type: str = 'float',
                      std_value: int = 3,
                      median_value: float = 0.023,
                      cap_zero: bool = True,
                      ddof: int = 1) -> Union[float, int]:
    """

    Calculate Percentile of a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param q: Percentile percent.
    :type q: float
    :return: Returns the percentile value.
    :rtype: float
    :example: *None*
    :note: If input values are floats, will return float values.

    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    if new_data.len == 0:
        return 0
    data_type = False
    if value_type == float:
        data_type = True
        temp_data = [item * 1000 for item in new_data.data]
    temp_data = _round_to(data=temp_data, val=1)
    ind = _round_to(data=new_data.len * q, val=1)
    temp_data.sort()
    for item in temp_data:
        if item >= temp_data[ind]:
            break
    if data_type:
        return item / 1000
    else:
        return item


def calc_gini(data: Union[list, np.ndarray, pd.Series],
              na_handling: str = 'none',
              value_type: str = 'float',
              std_value: int = 3,
              median_value: float = 0.023,
              cap_zero: bool = True,
              ddof: int = 1) -> Union[float, int]:
    """

    Calculate the Gini Coef for a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Gini value.
    :rtype: float
    :example:
        >>> lst = [4.3, 5.6]
        >>> calc_gini(data=lst, val=4, remainder=True) # 0.05445544554455435
    :note: The larger the gini coef, the more consolidated the chips on the table are to one person.
    """
    default = {'data': [], 'value_type': 'float', 'na_handling': 'none', 'std_value': 3, 'median_value': 0.023,
               'cap_zero': True, 'ddof': 1}
    new_data = Args(args=(locals(), default))
    # args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    # new_data = _prep_args(args=args)
    sorted_list = sorted(new_data.data)
    height, area = 0.0, 0.0
    for value in sorted_list:
        height += value
        area += height - value / 2.0
    fair_area = height * new_data.len / 2.0
    return _to_type(value=(fair_area - area) / fair_area, value_type=value_type)
