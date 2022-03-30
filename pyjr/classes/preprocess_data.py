"""
PreProcess class.

Usage:
 ./utils/preprocess_data.py

Author:
 Peter Rigali - 2022-03-23
"""
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
from pandas import DataFrame
from pyjr.classes.data import Data
from pyjr.utils.tools.math import _min, _max, _mean, _var, _std, _sum, _median, _mode, _skew, _kurtosis, _perc
from pyjr.utils.tools.clean import _mtype, _check_type
from sklearn.preprocessing import power_transform, quantile_transform, robust_scale


@dataclass
class PreProcess:

    __slots__ = ("cleanData", "data", "len", "name")

    def __init__(self, data: Optional[Data] = None):
        self.cleanData = data
        self.data = None
        self.len = None
        self.name = None

    def add_normalize(self, stat: str = 'min'):
        """Normalize data, default is min which keeps values between 0 and 1"""
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
        self.data = _check_type(d=lst, dtype=self.cleanData.dtype)
        self.len = self.data.__len__()
        self.name = self.cleanData.name + "_normalize_" + stat
        return self

    def add_standardize(self, stat: str = "mean"):
        """Standardize data, with a mean of 0 and std of 1"""
        if stat == "mean":
            lst = ((item - self.cleanData.mean) / self.cleanData.std for item in self.cleanData.data)
        elif stat == "median":
            temp_std = (self.cleanData.higher - self.cleanData.median)
            lst = ((item - self.cleanData.median) / temp_std for item in self.cleanData.data)
        self.data = _check_type(d=lst, dtype=self.cleanData.dtype)
        self.len = self.data.__len__()
        self.name = self.cleanData.name + "_standardize_" + stat
        return self

    def add_running(self,  window: int, stat: str = "mean", q: float = 0.50):
        """Calc running statistics"""
        calc = {"min": _min, "max": _max, "mean": _mean, "var": _var, "std": _std, "sum": _sum, "median": _median,
                "mode": _mode, "skew": _skew, "kurt": _kurtosis, "percentile": _perc}[stat]
        ran = range(window, self.cleanData.len)
        if stat != "percentile":
            self.name = self.cleanData.name + "_running_" + stat
            pre = [calc(data=self.cleanData.data[:window])] * window
            post = [calc(data=self.cleanData.data[i - window:i]) for i in ran]
        else:
            self.name = self.cleanData.name + "_running_" + stat + "_" + str(q)
            pre = [_perc(data=self.cleanData.data[:window], q=q)] * window
            post = [_perc(data=self.cleanData.data[i - window:i], q=q) for i in ran]
        self.data = _check_type(d=pre + post, dtype=self.cleanData.dtype)
        self.len = self.data.__len__()
        return self

    def add_cumulative(self, stat: str = "mean", q: float = 0.75):
        """Calc cumulative statistics"""
        calc = {"min": _min, "max": _max, "mean": _mean, "var": _var, "std": _std, "sum": _sum, "median": _median,
                "mode": _mode, "skew": _skew, "kurt": _kurtosis, "percentile": _perc}[stat]
        ran = range(1, self.cleanData.len)
        if stat != "percentile":
            self.name = self.cleanData.name + "_running_" + stat + "_" + str(q)
            lst = [0.0] + [calc(data=self.cleanData.data[:i]) for i in ran]
        else:
            self.name = self.cleanData.name + "_running_" + stat
            lst = [0.0] + [calc(data=self.cleanData.data[:i], q=q) for i in ran]
        self.data = _check_type(d=lst, dtype=self.cleanData.dtype)
        self.len = self.data.__len__()
        return self

    def add_log(self, constant: float = .01):
        lst = (math.log(i + constant) for i in self.cleanData.data)
        self.data = _check_type(d=lst, dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_log"
        self.len = self.data.__len__()
        return self

    def add_box_cox(self, lam: float = 0.1):
        """Only postive values"""
        """lambda = -1. is a reciprocal transform.
           lambda = -0.5 is a reciprocal square root transform.
           lambda = 0.0 is a log transform.
           lambda = 0.5 is a square root transform.
           lambda = 1.0 is no transform."""
        if self.cleanData is None:
            return self
        lst = ((i ** lam - 1) / lam for i in self.cleanData.data)
        self.data = _check_type(d=lst, dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_box_cox_" + str(lam)
        self.len = self.data.__len__()
        return self

    def add_sklearn_box_cox(self, standard: bool = True):
        """Only postive values"""
        arr = power_transform(X=self.cleanData.array(axis=1), method='box-cox', standardize=standard)
        self.data = _check_type(d=(i[0] for i in _mtype(d=arr)), dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_sklearn_box_cox"
        self.len = self.data.__len__()
        return self

    def add_sklearn_yeo_johnson(self, standard: bool = True):
        """Postive values and negative values"""
        arr = power_transform(X=self.cleanData.array(axis=1), method='yeo-johnson', standardize=standard)
        self.data = _check_type(d=(i[0] for i in _mtype(d=arr)), dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_sklearn_yeo_johnson"
        self.len = self.data.__len__()
        return self

    def add_sklearn_quantile(self, n_quantiles: int = 25, output_distribution: str = 'uniform'):
        """Recommended to not do before splitting"""
        """Also accepts 'normal' """
        arr = quantile_transform(X=self.cleanData.array(axis=1), n_quantiles=n_quantiles,
                                 output_distribution=output_distribution)
        self.data = _check_type(d=(i[0] for i in _mtype(d=arr)), dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_sklearn_quantile_" + str(n_quantiles) + "_" + output_distribution
        self.len = self.data.__len__()
        return self

    def add_sklearn_robust_scaling(self, with_centering: bool = True, with_scaling: bool = True,
                                   quantile_range: tuple = (25.0, 75.0)):
        """Recommended to not do before splitting"""
        arr = robust_scale(X=self.cleanData.array(axis=1), with_centering=with_centering, with_scaling=with_scaling,
                           quantile_range=quantile_range)
        self.data = _check_type(d=(i[0] for i in _mtype(d=arr)), dtype=self.cleanData.dtype)
        self.name = self.cleanData.name + "_sklearn_robust"
        self.len = self.data.__len__()
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
            name1 = 'constant'
            name2 = self.name
        self.data = arr
        self.name = (name1, name2)
        return self

    # Type Change Methods
    def list(self) -> list:
        """Returns a list"""
        if isinstance(self.data, (list, tuple)):
            return list(self.data)
        else:
            raise AttributeError("PreProcess data must be a list or tuple prior to this method call.")

    def tuple(self) -> tuple:
        """Returns a tuple"""
        if isinstance(self.data, (list, tuple)):
            return tuple(self.data)
        else:
            raise AttributeError("PreProcess data must be a list or tuple prior to this method call.")

    def array(self, axis: int = 0) -> np.ndarray:
        """Returns an np.ndarray"""
        if axis == 0:
            return np.array(self.data)
        else:
            return np.array(self.data).reshape(self.len, 1)

    def dataframe(self, index: list = None, name: str = None) -> DataFrame:
        """Returns a pd.DataFrame"""
        if index is None:
            index = range(self.len)

        if len(index) != self.len:
            raise AttributeError("List of index {} not equal to length of data {}.".format(len(index), self.len))

        if name is None and self.name is None:
            raise AttributeError("Need to pass a name.")
        else:
            if name is None:
                name = self.name
        return DataFrame(self.data, columns=[name], index=index)

    def dict(self) -> dict:
        """Returns a dict"""
        if isinstance(self.data, np.ndarray):
            return {self.name[ind]: self.data[:, ind] for ind, val in enumerate(self.data)}
        else:
            return {self.name: self.data}

    def __repr__(self):
        return 'PreProcessData'
