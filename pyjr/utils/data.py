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
import math
from scipy.stats import kstest, normaltest, shapiro, ttest_ind, ks_2samp
from pyjr.utils.base import _min, _max, _mean, _variance, _std, _sum, _median, _mode, _skew, _kurtosis, _percentile
from pyjr.utils.base import _percentiles, _unique_values, _check_na, _to_list, _check_list, _prep


@dataclass
class Data:
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

    __slots__ = ["name", "data", "len", "unique", "dtype", "mean", "median", "mode", "var", "std", "lower", "higher",
                 "min", "max", "sum", "skew", "kurt", "per", "distribution", "na"]

    def __init__(self, data, name: str = None, index = None, na_handling: str = 'none',
                 dtype: str = 'float', cap_zero: bool = True, std_value: int = 3, median_value: float = 0.023,
                 ddof: int = 1, q_lst: list = [0.159, 0.841], stats: bool = True,
                 distribution: bool = False):
        if name:
            self.name = name
        self.data = _prep(data=data, dtype=dtype, na_handling=na_handling, std_value=std_value,
                          median_value=median_value, cap_zero=cap_zero, ddof=ddof)
        if self.data is None:
            raise AttributeError("Data is empty.")

        self.unique = _unique_values(data=self.data, count=False)
        self.len = len(self.data)
        self.dtype = dtype

        if stats and dtype in {"float": True, "int": True}:
            self.mean = _mean(data=self.data)
            self.median = _median(data=self.data)
            self.mode = _mode(data=self.data)
            self.var = _variance(data=self.data, ddof=ddof)
            self.std = _std(data=self.data, ddof=ddof)
            self.lower, self.higher = _percentiles(data=self.data, q_lst=q_lst, dtype=self.dtype)
            self.min = _min(data=self.data)
            self.max = _max(data=self.data)
            self.sum = _sum(data=self.data)
            self.skew = _skew(data=self.data, length=self.len)
            self.kurt = _kurtosis(data=self.data, length=self.len)
            self.per = _percentile(data=self.data, q=0.75)
        else:
            self.mean, self.median, self.mode, self.var, self.std, self.lower, self.higher, self.min, self.max, self.sum, self.skew, self.kurt, self.per = None, None, None, None, None, None, None, None, None, None, None, None, None

        _na_ind_lst = [ind for ind, val in enumerate(_check_list(data=data)) if _check_na(value=val) == True]
        if len(_na_ind_lst) > 0:
            _percent_na = len(_na_ind_lst) / self.len
            self.na = {"index": _na_ind_lst, "percent": _percent_na}
        else:
            self.na = None

        if distribution:
            self.distribution, count = {'Kolmogorov-Smirnov': kstest(self.data, 'norm')[1],
                                        'DAgostino': normaltest(self.data)[1],
                                        'Shapiro-Wilk': shapiro(self.data)[1],
                                        'noraml': False}, 0
            for test_name in ('Kolmogorov-Smirnov', "DAgostino", 'Shapiro-Wilk'):
                if self.distribution[test_name] >= .05:
                    count += 1
            if count == 0:
                self.distribution['normal'] = True
        else:
            self.distribution = None

    # Comparison Methods
    # Distance
    def eucludian_distance(self, other) -> float:
        return math.dist(self.data, other.data) / self._len

    # MAPE
    # can not handle zero in denominator
    def mape(self, other) -> float:
        actual = np.array([i if i != 0.0 else .01 for i in self.data])
        pred = np.array([i if i != 0.0 else .01 for i in other.data])
        return np.mean(np.abs((actual - pred) / actual)) * 100

    # Auto-Correlation
    def auto_corr(self, other, num: int = 50) -> float:

        def acf(x, length):
            return [1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)]

        return np.corrcoef(acf(x=self.data, length=num), acf(x=other.data, length=num))[0, 1]

    # Correlation
    def corr(self, other) -> float:
        return np.corrcoef(self.data, other.data)[0, 1]

    # T Test to compare means
    def compare_means(self, other) -> float:
        return ttest_ind(a=self.data, b=other.data).pvalue

    # Kolmogorov-smirnov to see if from the same distribution
    def kol_smirnov(self, other) -> float:
        return ks_2samp(data1=self.data, data2=other.data).pvalue

    # Stat Methods
    def percentile(self, q: float) -> float:
        self.per = _percentile(data=self.data, q=q, dtype=self.dtype)
        return self.per

    # Type Change Methods
    def list(self) -> list:
        """Returns a list"""
        return list(self.data)

    def array(self, axis: int = 0) -> np.ndarray:
        """Returns an np.ndarray"""
        if axis == 0:
            return np.array(self.data)
        else:
            return np.array(self.data).reshape(self.len, 1)

    def dataframe(self, index: list = None, name: str = None) -> pd.DataFrame:
        """Returns a pd.DataFrame"""
        if index is None:
            index = range(self.len)

        if len(index) != self.len:
            raise AttributeError("List of index {} not equal to length of data {}.".format([len(index), self.len]))

        if name is None and self.name is None:
            raise AttributeError("Need to pass a name.")
        else:
            if name is None:
                name = self.name
        return pd.DataFrame(self.data, columns=[name], index=index)

    def __repr__(self):
        return 'CleanData'
