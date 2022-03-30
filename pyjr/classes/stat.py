"""
Stat class.

Usage:
 ./classes/stat.py

Author:
 Peter Rigali - 2022-03-30
"""
from dataclasses import dataclass
from pyjr.utils.base import _min, _max, _mean, _variance, _std, _sum, _median, _mode, _skew, _kurtosis, _percentile, _range
from pyjr.utils.tools import _check_na, _to_metatype, _replacement_value, _to_type, _empty


@dataclass
class Stat:
    """
    The Stat Class allows you to have preset data cleaning functions that are then applied to changing input data.

    :param stat: Desired stat to calculate.
    :type stat: str
    :param na: Desired handing of nan values.
    :type na: str
    :param dtype: Desired output dtype.
    :type dtype: str
    :param empty: If True and the dat is empty, will return 0.
    :type empty: bool
    """
    __slots__ = ('stat', 'na', 'dtype', 'empty')

    def __init__(self, stat: str, na: str = 'zero', dtype: str = 'float', empty: bool = False):
        self.stat = stat
        self.na = na
        self.dtype = dtype
        self.empty = empty

    def get(self, data, q: float = None):
        dic = {'mean': _mean, 'min': _min, 'max': _max, 'var': _variance, 'std': _std, 'sum': _sum, 'median': _median,
               'mode': _mode, 'skew': _skew, 'kurt': _kurtosis, 'percentile': _percentile, 'range': _range}
        data = _to_metatype(data=data, dtype='list')
        if self.empty:
            if _empty(data=data):
                return 0.0
        na = None
        for ind, val in enumerate(data):
            if _check_na(value=val):
                if na is None:
                    na = _replacement_value(data=data, na_handling=self.na)
                val = na
            data[ind] = val
        if q is not None and self.stat == 'percentile':
            return _percentile(data=data, q=q)
        else:
            return _to_type(value=dic[self.stat](data=data), dtype=self.dtype)

    def __repr__(self):
        return 'Stat'
