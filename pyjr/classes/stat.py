"""
Stat class.

Usage:
 ./classes/stat.py

Author:
 Peter Rigali - 2022-03-30
"""
from dataclasses import dataclass
from pyjr.utils.tools.math import _min, _max, _mean, _var, _std, _sum, _median, _mode, _skew, _kurtosis, _perc, _range
from pyjr.utils.tools.clean import _nan, _mtype, _rval, _type, _empty


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
        dic = {'mean': _mean, 'min': _min, 'max': _max, 'var': _var, 'std': _std, 'sum': _sum, 'median': _median,
               'mode': _mode, 'skew': _skew, 'kurt': _kurtosis, 'percentile': _perc, 'range': _range}
        data = _mtype(d=data, dtype='list')
        if self.empty:
            if _empty(d=data):
                return 0.0
        na = None
        for ind, val in enumerate(data):
            if _nan(v=val):
                if na is None:
                    na = _rval(d=data, na=self.na)
                val = na
            data[ind] = val
        if q is not None and self.stat == 'percentile':
            return _perc(data=data, q=q)
        else:
            return _type(v=dic[self.stat](data=data), dtype=self.dtype)

    def __repr__(self):
        return 'Stat'
