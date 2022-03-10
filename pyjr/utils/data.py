"""
CleanData class.

Usage:
 ./utils/data.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.stats import kstest, normaltest, shapiro


# Cleaning Functions
def _set_type(value, val_type: str = 'float'):
    """

    Sets the type of a value.

    :param value: Input Value
    :param val_type: Desired type.
    :type val_type: str
    :return: Value converted to desired type.
    :note: *None*
    """
    if val_type == 'float':
        if type(value) == float:
            return value
        else:
            return float(value)
    elif val_type == 'int':
        if type(value) == int:
            return value
        else:
            return int(value)
    elif val_type == 'str':
        if type(value) == str:
            return value
        else:
            return str(value)
    else:
        raise AttributeError('val_type not one of {float, int, str}.')


def _remove_na(data: list) -> list:
    """
    Remove Nan values from a list.

    :param data: List of data.
    :type data: list
    :return: Returns list with Nan values removed
    :rtype: list
    :note: Used internally
    """
    return [val for val in data if val == val and val is not None]


def _check_na(value, ind: int, na_ind_lst: list) -> bool:
    """
    Checks a value to see if Nan.

    :param value: Input value.
    :param ind: Input value index.
    :type ind: int.
    :param na_ind_lst: Existing Nan value indexes in a list.
    :type na_ind_lst: list.
    :return: Returns True or False if the value is Nan.
    :rtype: bool.
    :note: *None*
    """
    if value == value and value is not None:
        return False
    else:
        na_ind_lst.append(ind)
        return True


def _convert_na(value, replacement_val, val_lst: list, is_na: bool = False, val_type: str = 'float'):
    """

    Converts a Nan value to a desired value.

    :param value: Input value.
    :param replacement_val: Desired replacement value.
    :param val_lst: Existing list of values.
    :type val_lst: list.
    :param is_na: Input bool for if a value is Nan.
    :param val_type: Desired value type.
    :type val_type: str.
    :return: None, appends to existing list.
    :note: *None*
    """
    if is_na is False:
        val_lst.append(_set_type(value=value, val_type=val_type))
    else:
        val_lst.append(_set_type(value=replacement_val, val_type=val_type))


def _to_list(data) -> list:
    """
    Converts various inputs to a list.

    :param data: Input data.
    :return: A list.
    :rtype: list.
    :note: *None*
    """
    if type(data) == list:
        return data
    elif type(data) == pd.Series:
        return data.to_list()
    elif type(data) == np.ndarray:
        return data.tolist()
    return list(data)


def _unique_values(data: list, count: bool = False, order: bool = False, indexes: bool = False):
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
    if order:
        temp_dic, temp_lst = {}, []
        for item in data:
            if item not in temp_dic:
                temp_dic[item] = True
                temp_lst.append(item)
        return temp_lst
    if count:
        temp_data = list(set(data))
        return {i: data.count(i) for i in temp_data}
    if indexes:
        temp_dic, ind_dic = {}, {}
        for ind, item in enumerate(data):
            if item in temp_dic:
                ind_dic[item].append(ind)
            else:
                temp_dic[item] = True
                ind_dic[item] = [ind]
        return ind_dic
    return list(set(data))


def _search_dic_values(dic: dict, item):
    """
    Search a dictionary by the values.

    :param dic: Input dict.
    :type dic: dict.
    :param item: Desired value to search the dict for.
    :return: Key associated with the value.
    :note: *None*
    """
    return list(dic.keys())[list(dic.values()).index(item)]


def _round_to(data, val, remainder: bool = False, val_type: str = 'float'):
    """
    Rounds a value or list.

    :param data: Value or list.
    :param val: Place to round to.
    :param remainder: Whether to use remainder. If using floats, this should be true.
    :param val_type: Desired value type.
    :type val_type: str.
    :return: Returns a value or list of values.
    :note: *None*
    """
    if type(val) == int:
        val = float(val)

    if type(data) not in [list, pd.Series, np.ndarray]:
        if remainder is True:
            return round(_set_type(value=data, val_type=val_type) * val) / val
        else:
            return round(_set_type(value=data, val_type=val_type) / val) * val
    else:
        data = (_set_type(value=i, val_type=val_type) for i in data)
        if remainder is True:
            return [round(item * val) / val for item in data]
        else:
            return [round(item / val) * val for item in data]


# Internal Math Functions
def _max(data: list):
    """
    Find the max value of a list.

    :param data: Input data.
    :type data: list.
    :return: Maximum value.
    :note: *None*
    """
    if len(data) > 1:
        largest = 0
        for i in data:
            if i > largest:
                largest = i
        return largest
    elif len(data) == 0:
        return 0.0
    else:
        return data


def _min(data: list):
    """
    Find the min value of a list.

    :param data: Input data.
    :type data: list.
    :return: Minimum value.
    :note: *None*
    """
    if len(data) > 1:
        smallest = 0
        for i in data:
            if i < smallest:
                smallest = i
        return smallest
    elif len(data) == 0:
        return 0.0
    else:
        return data


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
    return sum((x - mu) ** 2 for x in data) / (len(data) - ddof)


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


def _sum(data: list):
    """
    Find the sum value of a list.

    :param data: Input data.
    :type data: list.
    :return: Sum value.
    :note: *None*
    """
    if len(data) > 1:
        return sum(data)
    elif len(data) == 0:
        return 0.0
    else:
        return data


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
        return sorted_lst[index]
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
        return float(first_val)
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
    nn = ((length * (length - 1)) ** .5) / (length - 2)
    return (((_sum(data=[i - mu for i in data]) ** 3) / length) / stdn) * nn


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


def _percentile(data: list, length: int, q: float, val_type: str = 'float'):
    """
    Find the percentile value of a list.

    :param data: Input data.
    :type data: list.
    :param length: Length of the input data.
    :type length: int.
    :param q: Percentile percent.
    :type q: float.
    :param val_type: Desired type.
    :type val_type: str.
    :return: Percentile value.
    :note: *None*
    """
    if length == 0:
        return 0
    data_type = False
    if val_type == 'float':
        data_type, data = True, [item * 1000 for item in data]
    data, ind = _round_to(data=data, val=1, val_type=val_type), _round_to(data=length * q, val=1, val_type=val_type)
    data.sort()
    for item in data:
        if item >= ind:
            break
    if data_type:
        return item / 1000
    else:
        return item


def _stats(data: list, length: int, ddof: int = 1):
    """
    Calculate various stats for a list.

    :param data: Input data.
    :type data: list.
    :param length: Length of input data.
    :type length: int.
    :param ddof: Desired Degrees of Freedom.
    :type ddof: int.
    :return: A group of stats.
    :note: *None*
    """
    return _min(data=data), _max(data=data), _mean(data=data), _variance(data=data, ddof=ddof), \
           _std(data=data, ddof=ddof), _sum(data=data), _median(data=data), _mode(data=data), \
           _skew(data=data, length=length), _kurtosis(data=data, length=length)


def _percentiles(data: list, length: int, q_lst: list = [0.159, 0.841], val_type: str = 'float'):
    """
    Calculate various percentiles for a list.

    :param data: Input data.
    :type data: list.
    :param length: Length of input data.
    :type length: int.
    :param q_lst: Desired percentile percents.
    :type q_lst: List of floats.
    :param val_type: Desired type.
    :type val_type: str.
    :return: A group of stats.
    :note: *None*
    """
    return (_percentile(data=data, length=length, q=q, val_type=val_type) for q in q_lst)


# Catch and replace na values
def _replacement_value(data: list, na_handling: str = 'median', std_val: int = 3, cap_zero: bool = True,
                       median_val: float = 0.023):
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
        return 0
    elif na_handling == 'mu':
        return _mean(data=_remove_na(data=data))
    elif na_handling == 'std':
        data_new = _remove_na(data=data)
        val = _mean(data=data_new) - (_std(data=data_new, ddof=1) * std_val)
        if cap_zero:
            if val > 0:
                return val
            else:
                return 0
        else:
            return val
    elif na_handling == 'median':
        return _percentile(data=data, length=len(data), q=median_val, val_type='float')


def _catch_na(data: list, na_handling: str = 'median', val_type: str = 'float', cap_zero: bool = True,
              std_value: int = 3, median_value: float = 0.023):
    """
    Catch and replace Nan values.

    :param data: Input data.
    :type data: list.
    :param na_handling: Desired Nan value handling method. {zero, mu, std, median}
    :type na_handling: str.
    :param val_type: Desired value type.
    :type val_type: str.
    :param cap_zero: Whether to cap the value at zero.
    :type cap_zero: bool.
    :param std_value: Desired Standard Deviation to use.
    :type std_value: int.
    :param median_value: Desired percentile to use.
    :type median_value: float.
    :return: Two lists
    :note: *None*
    """
    replacement_val = _set_type(value=_replacement_value(data=data, na_handling=na_handling, cap_zero=cap_zero,
                                                         std_val=std_value, median_val=median_value), val_type=val_type)
    val_lst, na_ind_lst = [], []
    for ind, val in enumerate(data):
        _convert_na(value=val, replacement_val=replacement_val, val_lst=val_lst,
                    is_na=_check_na(value=val, ind=ind, na_ind_lst=na_ind_lst), val_type=val_type)
    return val_lst, na_ind_lst


@dataclass
class CleanData:
    """

    Builds CleanData Class. Used for analysis of data.

    :param data: Input data.
    :type data:
    :param name: Input data name.
    :type name: str.
    :param index: Input data index.
    :type index:
    :param na_handling: Desired Nan value handling method. {zero, mu, std, median}
    :type na_handling: str.
    :param val_type: Desired type to fit data to.
    :type val_type: str.
    :param cap_zero: Whether to cap the value at zero.
    :type cap_zero: bool.
    :param std_value: Desired Standard Deviation to use.
    :type std_value: int.
    :param x: Whether the data is independent or not.
    type x: bool.
    :param y: Whether the data is dependent or not.
    type y: bool.
    :param ddof: Desired Degrees of Freedom.
    :type ddof: int.
    :param q_lst: List of columns to find the Quantile. *Optional*
    :type q_lst: list of floats.
    :example: *None*
    :note: *None*

    """
    def __init__(self, data, name: str, index, na_handling: str = 'median',
                 val_type: str = 'float', cap_zero: bool = True, std_value: int = 3, x: bool = True, y: bool = False,
                 ddof: int = 1, q_lst: list = [0.159, 0.841]):

        self._name = name
        self._na_handling = na_handling
        self._val_type = val_type
        self._cap_zero = cap_zero
        self._std_value = std_value
        self._x = x
        self._y = y
        self._ddof = ddof
        self._q_lst = q_lst
        self._new_data, self._na_ind_lst = _catch_na(data=_to_list(data=data), na_handling=na_handling,
                                                     val_type=val_type, cap_zero=cap_zero)
        self._len = len(self._new_data)
        if index is not None:
            if len(self._na_ind_lst) > 0:
                ind_dic = {i: True for i in self._na_ind_lst}
                temp_index_lst = []
                for ind, val in enumerate(_to_list(data=index)):
                    if ind not in ind_dic:
                        temp_index_lst.append(val)
                self._index_lst = temp_index_lst
            else:
                self._index_lst = _to_list(data=index)
        else:
            self._index_lst = list(range(self._len))

        self._min, self._max, self._mu, self._var, self._std, self._sum, self._median, self._mode, self._skew, self._kurt = _stats(data=self._new_data, length=self._len, ddof=self._ddof)
        self._lower, self._higher = _percentiles(data=self._new_data, length=self._len, q_lst=q_lst, val_type=val_type)
        self._unique_values = _unique_values(data=self._new_data)

        self._percent_na = 0.0
        if len(self._na_ind_lst) > 0:
            self._percent_na = len(self._na_ind_lst) / self._len

        self._is_normal = False
        self._dist_dict, count = {'Kolmogorov-Smirnov': kstest(self._new_data, 'norm')[1],
                                  "DAgostino": normaltest(self._new_data)[1],
                                  'Shapiro-Wilk': shapiro(self._new_data)[1]}, 0
        for test_name, test in self._dist_dict.items():
            if self._dist_dict[test_name] >= .05:
                count += 1
        if count == 0:
            self._is_normal = True

    def __getitem__(self):
        return self._new_data

    def __len__(self):
        return self._len

    def __repr__(self):
        return 'CleanData'

    @property
    def len(self) -> int:
        """Length of data"""
        return self._len

    @property
    def name(self) -> str:
        """Name of data"""
        return self._name

    @property
    def na_handling(self) -> str:
        """Method for handing Nan values"""
        return self._na_handling

    @property
    def value_type(self) -> str:
        """Desired type for data"""
        return self._val_type

    @property
    def cap_zero(self) -> bool:
        """Cap the lower value at 0"""
        return self._cap_zero

    @property
    def std_value(self) -> int:
        """Value to multiply the std by"""
        return self._std_value

    @property
    def is_x(self) -> bool:
        """Is the data Independent"""
        return self._x

    @property
    def is_y(self) -> bool:
        """is the data Dependent"""
        return self._y

    @property
    def ddof(self) -> int:
        """Degrees of Freedom used in std and var calcs"""
        return self._ddof

    @property
    def data(self) -> list:
        """Returned data"""
        return self._new_data

    @property
    def index(self) -> list:
        """Returned index"""
        return self._index_lst

    @property
    def na_ind_lst(self) -> list:
        """Indexes of Nan values"""
        return self._na_ind_lst

    @property
    def unique_values(self) -> list:
        """Unique Values in the data"""
        return self._unique_values

    @property
    def percent_na(self) -> float:
        """Percent of data that is Nan"""
        return self._percent_na

    @property
    def normal_tests(self) -> dict:
        """Dictionary of various normalcy tests"""
        return self._dist_dict

    @property
    def is_normal(self) -> bool:
        """If the data is normal"""
        return self._is_normal
