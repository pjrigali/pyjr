"""
Transformation functions.

Usage:
 ./utils/transformations.py

Author:
 Peter Rigali - 2022-03-10
"""
from typing import Union
import pandas as pd
import numpy as np
from pyjr.utils.base import  _check_type
from pyjr.utils.base import _min, _max, _mean, _std, _median, _percentile
from pyjr.utils.args import _prep_args


def normalize(data: Union[list, np.ndarray, pd.Series],
              na_handling: str = 'none',
              value_type: str = 'float',
              std_value: int = 3,
              median_value: float = 0.023,
              cap_zero: bool = True,
              ddof: int = 1) -> list:
    """

    Normalize a list between 0 and 1.

    :param data: Input data to normalize.
    :type data: list, np.ndarray, or pd.Series
    :param keep_nan: If True, will maintain nan values, default is False. *Optional*
    :type keep_nan: bool
    :return: Normalized list.
    :rtype: list
    :example: *None*
    :note: If an int or float is passed for keep_nan, that value will be placed where nan's are present.

    """
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    data = _prep_args(args=args)
    max_val, min_val = _max(data), _min(data)
    max_min_val = max_val - min_val
    if max_min_val == 0.0:
        max_min_val = 1.0
    return _check_type(data=[(val - min_val) / max_min_val for val in temp_data], value_type=value_type)


def standardize(data: Union[list, np.ndarray, pd.Series],
                na_handling: str = 'none',
                value_type: str = 'float',
                std_value: int = 3,
                median_value: float = 0.023,
                cap_zero: bool = True,
                ddof: int = 1) -> list:
    """

    Standardize a list with a mean of zero and std of 1.

    :param data: Input data to standardize.
    :type data: list, np.ndarray, or pd.Series
    :param keep_nan: If True, will maintain nan values, default is False. *Optional*
    :type keep_nan: bool
    :return: Standardized list.
    :rtype: list
    :example: *None*
    :note: If an int or float is passed for keep_nan, that value will be placed where nan's are present.

    """
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    data = _prep_args(args=args)
    mu = _mean(data=temp_data)
    if _std(data=temp_data, ddof=1) != 0:
        return _check_type(data=[(item - mu) / std for item in data], value_type=value_type)
    else:
        return _check_type(data=[0] * len(data), value_type=value_type)


def running_mean(data: Union[list, np.ndarray, pd.Series],
                 num: int,
                 na_handling: str = 'mean',
                 value_type: str = 'float',
                 std_value: int = 3,
                 median_value: float = 0.023,
                 cap_zero: bool = True,
                 ddof: int = 1) -> list:
    """

    Calculate the Running Mean on *num* interval.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param num: Input val used for running mean.
    :type num: int
    :return: Running mean for a given  np.ndarray, pd.Series, or list.
    :rtype: List[float]
    :example: *None*
    :note: None and np.nan values are replaced with the mean value.

    """
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    data = _prep_args(args=args)
    pre = ([_mean(data=data[:num])] * num) + [_mean(data=data[i - num:i]) for i in range(num, len(data))]
    return _check_type(data=pre, value_type=value_type)


def running_std(data: Union[list, np.ndarray, pd.Series],
                 num: int,
                 na_handling: str = 'mean',
                 value_type: str = 'float',
                 std_value: int = 3,
                 median_value: float = 0.023,
                 cap_zero: bool = True,
                 ddof: int = 1) -> list:
    """

    Calculate the Running Std on *num* interval.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param num: Input val used for Running Std window.
    :type num: int
    :return: Running std for a given  np.ndarray, pd.Series, or list.
    :rtype: List[float]
    :example: *None*
    :note: None and np.nan values are replaced with the mean value.

    """
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    data = _prep_args(args=args)
    pre = ([_std(data=data[:num])] * num) + [_std(data=data[i - num:i]) for i in range(num, len(data))]
    return _check_type(data=pre, value_type=value_type)


def running_median(data: Union[list, np.ndarray, pd.Series],
                   num: int,
                   na_handling: str = 'mean',
                   value_type: str = 'float',
                   std_value: int = 3,
                   median_value: float = 0.023,
                   cap_zero: bool = True,
                   ddof: int = 1) -> list:
    """

    Calculate the Running Median on *num* interval.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param num: Input val used for Running median window.
    :type num: int
    :return: list.
    :rtype: List[float]
    :example: *None*
    :note: None and np.nan values are replaced with the mean value.

    """
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    data = _prep_args(args=args)
    pre = ([_median(data=data[:num])] * num) + [_median(data=data[i - num:i]) for i in  range(num, len(data))]
    return _check_type(data=pre, value_type=value_type)


def running_percentile(data: Union[list, np.ndarray, pd.Series],
                       num: int,
                       q: float,
                       na_handling: str = 'median',
                       value_type: str = 'float',
                       std_value: int = 3,
                       median_value: float = 0.023,
                       cap_zero: bool = True,
                       ddof: int = 1) -> list:
    """

    Calculate the Running Percentile on *num* interval.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :param num: Input val used for Running Percentile window.
    :type num: int
    :param q: Percent of data.
    :type q: float
    :return: Running percentile for a given  np.ndarray, pd.Series, or list.
    :rtype: List[float]
    :example: *None*
    :note: None and np.nan values are replaced with the mean value.

    """
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    data = _prep_args(args=args)
    ran = range(num, len(data))
    pre = ([_percentile(data=data[:num], q=q)] * num) + [_percentile(data=data[i - num:i], q=q) for i in ran]
    return _check_type(data=pre, value_type=value_type)


def cumulative_mean(data: Union[list, np.ndarray, pd.Series],
                    na_handling: str = 'median',
                    value_type: str = 'float',
                    std_value: int = 3,
                    median_value: float = 0.023,
                    cap_zero: bool = True,
                    ddof: int = 1) -> list:
    """

    Calculate the Cumulative Mean.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Cumulative mean for a given np.ndarray, pd.Series, or list.
    :rtype: List[float]
    :example: *None*
    :note: None and np.nan values are replaced with the mean value.

    """
    args = (data, value_type, na_handling, std_value, median_value, cap_zero, ddof)
    data = _prep_args(args=args)
    return _check_type(data=[0.0] + [_mean(data=data[:i]) for i in range(1, len(data))], value_type=value_type)
