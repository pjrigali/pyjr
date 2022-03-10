"""
CleanData class.

Usage:
 ./utils/cleandata.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.stats import kstest, normaltest, shapiro
from pyjr.utils.base import _min, _max, _mean, _variance, _std, _sum, _median, _mode, _skew, _kurtosis, _percentile
from pyjr.utils.base import _percentiles, _unique_values, _check_na, _to_list, _check_list, _prep


# def _stats(data: list, length: int, ddof: int = 1):
#     """
#     Calculate various stats for a list.
#
#     :param data: Input data.
#     :type data: list.
#     :param length: Length of input data.
#     :type length: int.
#     :param ddof: Desired Degrees of Freedom.
#     :type ddof: int.
#     :return: A group of stats.
#     :note: *None*
#     """
#     return _min(data=data), _max(data=data), _mean(data=data), _variance(data=data, ddof=ddof), \
#            _std(data=data, ddof=ddof), _sum(data=data), _median(data=data), _mode(data=data), \
#            _skew(data=data, length=length), _kurtosis(data=data, length=length)


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
    def __init__(self, data, name: str, index = None, na_handling: str = 'none',
                 value_type: str = 'float', cap_zero: bool = True, std_value: int = 3, median_value: float = 0.023,
                 x: bool = True, y: bool = False, ddof: int = 1, q_lst: list = [0.159, 0.841]):

        self._name = name
        # self._na_handling = na_handling
        # self._value_type = value_type
        # self._cap_zero = cap_zero
        # self._std_value = std_value
        # self._median_value = median_value
        self._x = x
        self._y = y
        self._ddof = ddof
        # self._q_lst = q_lst
        self._inputs = {'data': data, 'value_type': value_type, 'na_handling': na_handling, 'std_value': std_value,
                        'median_value': median_value, 'cap_zero': cap_zero, 'ddof': ddof, 'q_lst': q_lst}
        self._new_data = _prep(data=data, value_type=value_type, na_handling=na_handling, std_value=std_value,
                               median_value=median_value, cap_zero=cap_zero,ddof=ddof)
        self._na_ind_lst =[ind for ind, val in enumerate(_check_list(data=data)) if _check_na(value=val) == True]
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

        # self._min, self._max, self._mu, self._var, self._std, self._sum, self._median, self._mode, self._skew, \
        # self._kurt = _stats(data=self._new_data, length=self._len, ddof=self._ddof)
        self._lower, self._higher = _percentiles(data=self._new_data, length=self._len, q_lst=q_lst, value_type=value_type)
        self._unique_values = _unique_values(data=self._new_data, count=False)

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

    # @property
    # def na_handling(self) -> str:
    #     """Method for handing Nan values"""
    #     return self._na_handling
    #
    # @property
    # def value_type(self) -> str:
    #     """Desired type for data"""
    #     return self._value_type
    #
    # @property
    # def cap_zero(self) -> bool:
    #     """Cap the lower value at 0"""
    #     return self._cap_zero
    #
    # @property
    # def std_value(self) -> int:
    #     """Value to multiply the std by"""
    #     return self._std_value
    #
    # @property
    # def median_value(self) -> int:
    #     """Value to multiply the std by"""
    #     return self._median_value

    @property
    def is_x(self) -> bool:
        """Is the data Independent"""
        return self._x

    @property
    def is_y(self) -> bool:
        """is the data Dependent"""
        return self._y

    # @property
    # def ddof(self) -> int:
    #     """Degrees of Freedom used in std and var calcs"""
    #     return self._ddof

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

    @property
    def min(self):
        return _min(self._new_data)

    @property
    def max(self):
        return _max(self._new_data)

    @property
    def mean(self):
        return _mean(self._new_data)

    @property
    def var(self):
        return _variance(self._new_data, ddof=self._ddof)

    @property
    def std(self):
        return  _std(self._new_data, ddof=self._ddof)

    @property
    def sum(self):
        return _sum(self._new_data)

    @property
    def median(self):
        return _median(self._new_data)

    @property
    def mode(self):
        return _mode(self._new_data)

    @property
    def skew(self):
        return _skew(self._new_data, length=self._len)

    @property
    def kurt(self):
        return _kurtosis(self._new_data, length=self._len)

    @property
    def lower_percentile(self):
        return self._lower

    @property
    def higher_percentile(self):
        return self._higher

    @property
    def inputs(self):
        return self._inputs
