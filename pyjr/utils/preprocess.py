from dataclasses import dataclass
from typing import Union
import math
import numpy as np
from pyjr.utils.data import Data
from pyjr.utils.base import _check_type, _check_list, _min, _max, _mean, _variance, _std, _sum, _median, _mode, _skew, _kurtosis, _percentile, _unique_values
from sklearn.preprocessing import power_transform, quantile_transform, robust_scale


def one_hot_encode(data: Union[list, tuple]) -> tuple:
    """One hot encode a list of data"""
    unique = _unique_values(data=data, count=False)
    clean_data_lst = []
    for key in unique:
        clean_data_lst.append(Data(data=[1.0 if key == val else 0.0 for val in data], name=key + '_ohe'))
    return tuple(clean_data_lst)


@dataclass
class PreProcess:

    __slots__ = ["cleanData", "data", "len", "name"]

    def __init__(self, data: Data):
        self.cleanData = data
        self.data = None
        self.len = None
        self.name = None

    def normalize(self, stat: str = 'min'):
        """Normalize data, default is min which keeps values between 0 and 1"""
        if self.cleanData.max is None and self.cleanData.min is None:
            self.cleanData.max = _max(data=self.cleanData.data)
            self.cleanData.min = _min(data=self.cleanData.data)
        max_min_val = self.cleanData.max - self.cleanData.min
        if max_min_val == 0.0:
            max_min_val = 1.0
        if stat == "mean":
            lst = ((val - self.cleanData.mean) / max_min_val for val in self.cleanData.data)
        elif stat == "min":
            lst = ((val - self.cleanData.min) / max_min_val for val in self.cleanData.data)
        elif stat == "median":
            lst = ((val - self.cleanData.median) / max_min_val for val in self.cleanData.data)
        else:
            raise AttributeError('Stat must be (mean, min, median)')
        self.data = _check_type(data=lst, dtype=self.cleanData.dtype)
        self.len = len(self.data)
        self.name = self.cleanData.name + "_normalize_" + stat
        return self

    def standardize(self, stat: str = "mean"):
        """Standardize data, with a mean of 0 and std of 1"""
        if stat == "mean":
            if self.cleanData.mean is None:
                self.cleanData.mean = _mean(data=self.cleanData.data)
                self.cleanData.std = _std(data=self.cleanData.data)
            lst = ((item - self.cleanData.mean) / self.cleanData.std for item in self.cleanData.data)
        elif stat == "median":
            if self.cleanData.median is None:
                self.cleanData.median = _median(data=self.cleanData.data)
                self.cleanData.higher = _percentile(data=self.cleanData.data, q= 0.841)
            temp_std = (self.cleanData.higher - self.cleanData.median)
            lst = ((item - self.cleanData.median) / temp_std for item in self.cleanData.data)
        self.data = _check_type(data=lst, dtype=self.cleanData.dtype)
        self.len = len(self.data)
        self.name = self.cleanData.name + "_standardize_" + stat
        return self

    def running(self,  window: int, stat: str = "mean", q: float = 0.50):
        """Calc running statistics"""
        calc = {"min": _min, "max": _max, "mean": _mean, "var": _variance, "std": _std, "sum": _sum, "median":_median,
                "mode": _mode, "skew": _skew, "kurt": _kurtosis, "percentile": _percentile}[stat]
        ran = range(window, self.cleanData.len)
        if stat != "percentile":
            self.name = self.cleanData.name + "_running_" + stat
            pre = [calc(data=self.cleanData.data[:window])] * window
            post = [calc(data=self.cleanData.data[i - window:i]) for i in ran]
        else:
            self.name = self.cleanData.name + "_running_" + stat + "_" + str(q)
            pre = [_percentile(data=self.cleanData.data[:window], q=q)] * window
            post = [_percentile(data=self.cleanData.data[i - window:i], q=q) for i in ran]
        self.data = _check_type(data=pre + post, dtype=self.cleanData.dtype)
        self.len = len(self.data)
        return self

    def cumulative(self, stat: str = "mean", q: float = 0.75):
        """Calc cumulative statistics"""
        calc = {"min": _min, "max": _max, "mean": _mean, "var": _variance, "std": _std, "sum": _sum, "median":_median,
                "mode": _mode, "skew": _skew, "kurt": _kurtosis, "percentile": _percentile}[stat]
        ran = range(1, self.cleanData.len)
        if stat != "percentile":
            self.name = self.cleanData.name + "_running_" + stat + "_" + str(q)
            lst = [0.0] + [calc(data=self.cleanData.data[:i]) for i in ran]
        else:
            self.name = self.cleanData.name + "_running_" + stat
            lst = [0.0] + [calc(data=self.cleanData.data[:i], q=q) for i in ran]
        self.data = _check_type(data=lst, dtype=self.cleanData.dtype)
        self.len = len(self.data)
        return self

    def log(self, constant: float = .01):
        lst = (math.log(i + constant) for i in self.cleanData.data)
        self.data = _check_type(data=lst, dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_log"
        self.len = len(self.data)
        return self

    def box_cox(self, lam: float = 0.1):
        """Only postive values"""
        """lambda = -1. is a reciprocal transform.
           lambda = -0.5 is a reciprocal square root transform.
           lambda = 0.0 is a log transform.
           lambda = 0.5 is a square root transform.
           lambda = 1.0 is no transform."""
        lst = ((i ** lam - 1) / lam for i in self.cleanData.data)
        self.data = _check_type(data=lst, dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_box_cox_" + str(lam)
        self.len = len(self.data)
        return self

    def sklearn_box_cox(self, standard: bool = True):
        """Only postive values"""
        arr = power_transform(X=self.cleanData.array(axis=1), method='box-cox', standardize=standard)
        self.data = _check_type(data=(i[0] for i in _check_list(data=arr)), dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_sklearn_box_cox"
        self.len = len(self.data)
        return self

    def sklearn_yeo_johnson(self, standard: bool = True):
        """Postive values and negative values"""
        arr = power_transform(X=self.cleanData.array(axis=1), method='yeo-johnson', standardize=standard)
        self.data = _check_type(data=(i[0] for i in _check_list(data=arr)), dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_sklearn_yeo_johnson"
        self.len = len(self.data)
        return self

    def sklearn_quantile(self, n_quantiles: int = 25, output_distribution: str = 'uniform'):
        """Recommended to not do before splitting"""
        """Also accepts 'normal' """
        arr = quantile_transform(X=self.cleanData.array(axis=1), n_quantiles=n_quantiles,
                                 output_distribution=output_distribution)
        self.data = _check_type(data=(i[0] for i in _check_list(data=arr)), dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_sklearn_quantile_" + str(n_quantiles) + "_" + output_distribution
        self.len = len(self.data)
        return self

    def sklearn_robust_scaling(self, with_centering: bool = True, with_scaling: bool = True,
                               quantile_range: tuple = (25.0, 75.0)):
        """Recommended to not do before splitting"""
        arr = robust_scale(X=self.cleanData.array(axis=1), with_centering=with_centering, with_scaling=with_scaling,
                           quantile_range=quantile_range)
        self.data = _check_type(data=(i[0] for i in _check_list(data=arr)), dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_sklearn_robust"
        self.len = len(self.data)
        return self

    def add_constant(self, columns: int = 2, other: Data = None):
        """Adds a column of 1's to the data"""
        if other:
            if other.len != self.cleanData.len:
                if other.name is None:
                    name1 = "First Array"
                else:
                    name1 = other.name
                if self.cleanData.name is None:
                    name2 = "Second Array"
                else:
                    name2 = self.cleanData.name
                raise AttributeError("Lengths of the two Data's are different: " + name1 + " {}, " + name2 + " {})".format((other.len, self.cleanData.len)))
            else:
                arr = np.ones((self.cleanData.len, 2))
                arr[:, 0] = self.cleanData.array(axis=0)
                arr[:, 1] = other.array(axis=0)
        else:
            arr = np.ones((self.cleanData.len, columns))
            arr[:, 1] = self.data
        self.data = arr
        return self

    def __repr__(self):
        return 'PreProcessData'
