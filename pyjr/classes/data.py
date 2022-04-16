"""
CleanData class.

Usage:
 ./utils/data.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
from pandas import DataFrame
import numpy as np
from scipy.stats import kstest, normaltest, shapiro
from pyjr.utils.tools.math import _min, _max, _mean, _var, _std, _sum, _med, _mod, _skew, _kurt, _perc, _percs, _range
from pyjr.utils.tools.math import _gini
from pyjr.utils.tools.clean import _nan, _prep, _mtype
from pyjr.utils.tools.general import _unique_values


@dataclass
class Data:
    """

    Builds CleanData Class. Used for analysis of data.

    :param data: Input data.
    :type data:
    :param name: Input data name.
    :type name: str.
    :param na_handling: Desired Nan value handling method. {zero, mu, std, median}
    :type na_handling: str.
    :param dtype: Desired type to fit data to.
    :type dtype: str.
    :param cap_zero: Whether to cap the value at zero.
    :type cap_zero: bool.
    :param std_value: Desired Standard Deviation to use.
    :type std_value: int.
    :param ddof: Desired Degrees of Freedom.
    :type ddof: int.
    :param q_lst: List of columns to find the Quantile. *Optional*
    :type q_lst: list of floats.
    :example: *None*
    :note: *None*

    """

    __slots__ = ("name", "data", "len", "unique", "dtype", "mean", "median", "mode", "var", "std", "lower", "higher",
                 "min", "max", "sum", "skew", "kurt", "per", "rang", "distribution", "na", "gini")

    def __init__(self, data = None, name: str = None, na_handling: str = 'none',
                 dtype: str = 'float', cap_zero: bool = True, std_value: int = 3, median_value: float = 0.023,
                 ddof: int = 1, q_lst: tuple = (0.159, 0.841), stats: bool = True,
                 distribution: bool = False, unique: bool = False):

        self.name = None
        if name:
            self.name = name
        self.data = _prep(d=data, dtype=dtype, na=na_handling)
        self.unique = None
        if unique:
            self.unique = _unique_values(data=self.data, count=False)

        self.len = self.data.__len__()
        self.dtype = dtype
        if stats and dtype in {"float": True, "int": True}:
            self.mean = _mean(d=self.data)
            self.median = _med(d=self.data)
            self.mode = _mod(d=self.data)
            self.var = _var(d=self.data, dof=ddof)
            self.std = _std(d=self.data, dof=ddof)
            self.lower, self.higher = _percs(d=self.data, q_lst=q_lst)
            self.min = _min(d=self.data)
            self.max = _max(d=self.data)
            self.sum = _sum(d=self.data)
            self.skew = _skew(d=self.data)
            self.kurt = _kurt(d=self.data)
            self.per = _perc(d=self.data, q=0.75)
            self.rang = _range(d=self.data)
            self.gini = _gini(d=self.data)
        else:
            self.mean, self.median, self.mode, self.var, self.std = None, None, None, None, None
            self.lower, self.higher, self.min, self.max, self.sum = None, None, None, None, None
            self.skew, self.kurt, self.per, self.rang, self.gini = None, None, None, None, None

        self.na = None
        if self.data.__len__() != data.__len__():
            tup = _mtype(d=data, dtype='tuple')
            _na_ind_lst = [ind for ind, val in enumerate(tup) if _nan(v=val) == True]
            if _na_ind_lst.__len__() > 0:
                _percent_na = _na_ind_lst.__len__() / self.len
                self.na = {"index": _na_ind_lst, "percent": _percent_na}

        self.distribution = None
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

    def add_percentile(self, q: float) -> float:
        self.per = _perc(d=self.data, q=q)
        return self.per

    # Type Change Methods
    def astype(self, dtype):
        self.data = tuple([dtype(i) for i in self.data])
        self.dtype = dtype.__str__()
        return self

    def list(self) -> list:
        """Returns a list"""
        return list(self.data)

    def tuple(self) -> tuple:
        """Returns a tuple"""
        return tuple(self.data)

    def array(self, axis: int = 0, ts: bool = False) -> np.ndarray:
        """Returns an np.ndarray"""
        if ts:
            arr = np.ones((self.data.__len__(), 2))
            arr[:, 0] = self.data
            arr[:, 1] = list(range(self.data.__len__()))
            return arr.reshape((2, self.data.__len__()))
        else:
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
        return {self.name: self.data}

    def __repr__(self):
        return 'CleanData'
