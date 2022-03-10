"""
Stand-alone functions.

Usage:
 ./utils/standalone.py

Author:
 Peter Rigali - 2022-03-10
"""
from typing import Union
from base import _prep, _prep_args, _unique_values, _to_list, _search_dic_values, _to_type, _max
from args import native_mean_args, native_median_args, native_variance_args, native_std_args, native_sum_args


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
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        if order:
            temp_dic, temp_lst = {}, []
            for item in data:
                if item not in temp_dic:
                    temp_dic[item] = True
                    temp_lst.append(item)
            return temp_lst
        if count:
            return {i: data.count(i) for i in set(data)}
        if indexes:
            temp_dic, ind_dic = {}, {}
            for ind, item in enumerate(data):
                if item in temp_dic:
                    ind_dic[item].append(ind)
                else:
                    temp_dic[item] = True
                    ind_dic[item] = [ind]
            return ind_dic
        return _to_list(data=set(data))


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
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        count_dic = unique_values(data=data, count=True)
        count_dic_values = _to_list(data=count_dic.values())
        lst, dic_max = [], _max(count_dic_values)
        for i in count_dic_values:
            lst.append((_search_dic_values(dic=count_dic, item=dic_max), i))
            count_dic_values = _to_list(data=count_dic.values())
            continue
            # val = search_dic_values(dic=count_dic, item=dic_max)
            # lst.append((val, i))
            # del count_dic[val]
            # count_dic_values = _to_list(data=count_dic.values())

        first_val, second_val = lst[0][0], lst[0][1]
        with [i[0] for i in lst if second_val == i[1]] as equal_lst:
            new_args = (equal_lst, value_type, na_handling, std_value, median_value, cap_zero, ddof)
            with len(equal_lst) as lst_len:
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
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        with sorted(data) as sorted_lst:
            with len(data) as lst_len:
                with (lst_len - 1) // 2 as index:
                    if lst_len % 2:
                        return _to_type(value=sorted_lst[index], value_type=value_type)
                    else:
                        new_data = [sorted_lst[index]] + [sorted_lst[index + 1]]
                        new_args = (new_data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
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
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        with len(data) as lst_len:
            if lst_len != 0:
                return _to_type(value=sum(data) / lst_len, value_type=value_type)
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
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        new_args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
        with native_mean_args(args=new_args) as mu:
            return _to_type(value=sum((x - mu) ** 2 for x in data) / (len(data) - ddof), value_type=value_type)


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
    data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
                 median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
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
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        with len(data) as lst_len:
            if lst_len > 1:
                return sum(data)
            elif lst_len == 0:
                return 0.0
            else:
                return data


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
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        with len(data) as lst_len:
            if lst_len > 1:
                largest = 0.0
                for i in data:
                    if i > largest:
                        largest = i
                return largest
            elif lst_len == 0:
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
    # data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
    #              median_value=median_value, cap_zero=cap_zero, ddof=ddof)
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        with len(data) as lst_len:
            if lst_len > 1:
                smallest = 0.0
                for i in data:
                    if i < smallest:
                        smallest = i
                return smallest
            elif lst_len == 0:
                return 0.0
            else:
                return data

# def unique_values(data: Union[list, np.ndarray, pd.Series], count: Optional[bool] = None, order: Optional[bool] = None,
#                   indexes: Optional[bool] = None, keep_nan: Optional[bool] = False) -> Union[list, dict]:
#     """
#
#     Get Unique values from a list.
#
#     :param data: Input data.
#     :type data: list, np.ndarray, or pd.Series
#     :param count: Return a dictionary with item and count, default is None. *Optional*
#     :type count: bool
#     :param order: If True will maintain the order, default is None. *Optional*
#     :type order: bool
#     :param indexes: If True will return index of all similar values, default is None. *Optional*
#     :type indexes: bool
#     :param keep_nan: If True will keep np.nan and None values, converting them to None, default is False. *Optional*
#     :type keep_nan: bool
#     :return: Returns either a list of unique values or a dict of unique values with counts.
#     :rtype: Union[list, dict]
#     :example: *None*
#     :note: Ordered may not appear accurate if viewing in IDE.
#
#     """
#     data = _remove_nan(data=_to_list(data=data), keep_nan=keep_nan)
#
#     if order:
#         temp_dic, temp_lst = {}, []
#         for item in data:
#             if item not in temp_dic:
#                 temp_dic[item] = True
#                 temp_lst.append(item)
#         return temp_lst
#     if count:
#         temp_data = list(set(data))
#         return {i: data.count(i) for i in temp_data}
#     if indexes:
#         temp_dic, ind_dic = {}, {}
#         for ind, item in enumerate(data):
#             if item in temp_dic:
#                 ind_dic[item].append(ind)
#             else:
#                 temp_dic[item] = True
#                 ind_dic[item] = [ind]
#         return ind_dic
#     return list(set(data))


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
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        temp_args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
        with len(data) as n:
            mu, stdn = native_mean_args(args=temp_args), native_std_args(args=temp_args) ** 3
            new_args = ([i - mu for i in data], value_type, na_handling, std_value, median_value, cap_zero, ddof)
            return (((native_sum_args(args=new_args) ** 3) / n) / stdn) * ((n * (n-1))**.5) / (n - 2)


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
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        temp_args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
        mu, stdn = native_mean_args(args=temp_args), native_std_args(args=args) ** 4
        new_args = ([i - mu for i in data], value_type, na_handling, std_value, median_value, cap_zero, ddof)
        return (((native_sum_args(args=new_args) ** 4) / len(data)) / stdn) - 3


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
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        if len(data) == 0:
            return 0
        data_type = False
        if type(data[0]) == float:
            data_type = True
            data = [item * 1000 for item in data]
        data = round_to(data=data, val=1)
        ind = round_to(data=len(data) * q, val=1)
        data.sort()
        for item in data:
            if item >= ind:
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
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    with _prep_args(args=args) as data:
        sorted_list = sorted(data)
        height, area = 0.0, 0.0
        for value in sorted_list:
            height += value
            area += height - value / 2.0
        fair_area = height * len(data) / 2.0
        return _to_type(value=(fair_area - area) / fair_area, value_type=value_type)