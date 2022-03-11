from dataclasses import dataclass
import math
import numpy as np
from pyjr.utils.cleandata import CleanData
from pyjr.utils.base import _check_type, _mean, _percentile, _median, _std, _check_list
from sklearn.preprocessing import power_transform, quantile_transform, robust_scale

@dataclass
class TransformData:

    def __init__(self, data: CleanData):
        self._cleanData = data
        self._value_type = self._cleanData.inputs['value_type']
        self._data = self._cleanData.data
        self._len = self._cleanData.len
        self._ddof = self._cleanData.inputs['ddof']

    def normalize_minmax(self) -> list:
        max_min_val = self._cleanData.max - self._cleanData.min
        if max_min_val == 0.0:
            max_min_val = 1.0
        lst = ((val - self._cleanData.min) / max_min_val for val in self._data)
        return _check_type(data=lst, value_type=self._value_type)

    def normalize_mean(self) -> list:
        max_min_val = self._cleanData.max - self._cleanData.min
        if max_min_val == 0.0:
            max_min_val = 1.0
        lst = ((val - self._cleanData.mean) / max_min_val for val in self._data)
        return _check_type(data=lst, value_type=self._value_type)

    def normalize_median(self) -> list:
        max_min_val = self._cleanData.max - self._cleanData.min
        if max_min_val == 0.0:
            max_min_val = 1.0
        lst = ((val - self._cleanData.median) / max_min_val for val in self._data)
        return _check_type(data=lst, value_type=self._value_type)

    def standardize(self) -> list:
        lst = ((item - self._cleanData.mean) / self._cleanData.std for item in self._data)
        return _check_type(data=lst, value_type=self._value_type)

    def standardize_median(self) -> list:
        lst = ((item - self._cleanData.median) / self._cleanData.lower_percentile for item in self._data)
        return _check_type(data=lst, value_type=self._value_type)

    def running_mean(self, num: int) -> list:
        ran = range(num, self._len)
        pre = [_mean(data=self._data[:num])] * num
        post = [_mean(data=self._data[i - num:i]) for i in ran]
        return _check_type(data=pre + post, value_type=self._value_type)

    def running_std(self, num: int) -> list:
        ran = range(num, self._len)
        pre = [_std(data=self._data[:num], ddof=self._ddof)] * num
        post = [_std(data=self._data[i - num:i], ddof=self._ddof) for i in ran]
        return _check_type(data=pre + post, value_type=self._value_type)

    def running_median(self, num: int) -> list:
        ran = range(num, self._len)
        pre = [_median(data=self._data[:num])] * num
        post = [_median(data=self._data[i - num:i]) for i in ran]
        return _check_type(data=pre + post, value_type=self._value_type)

    def running_percentile(self, num: int, q: float) -> list:
        ran = range(num, self._len)
        pre = [_percentile(data=self._data[:num], q=q)] * num
        post = [_percentile(data=self._data[i - num:i], q=q) for i in ran]
        return _check_type(data=pre + post, value_type=self._value_type)

    def cumulative_mean(self) -> list:
        ran = range(1, self._len)
        pre = [0.0]
        post = [_mean(data=self._data[:i]) for i in ran]
        return _check_type(data=pre + post, value_type=self._value_type)

    def log(self, constant: float = .01) -> list:
        lst = (math.log(i + constant) for i in self._data)
        return _check_type(data=lst, value_type=self._value_type)

    def box_cox(self, lam: float = 0.1) -> list:
        """Only postive values"""
        """lambda = -1. is a reciprocal transform.
           lambda = -0.5 is a reciprocal square root transform.
           lambda = 0.0 is a log transform.
           lambda = 0.5 is a square root transform.
           lambda = 1.0 is no transform."""
        lst = ((i ** lam - 1) / lam for i in self._data)
        return _check_type(data=lst, value_type=self._value_type)

    def sklearn_box_cox(self, standard: bool = True):
        """Only postive values"""
        arr = power_transform(X=np.array(self._data).reshape(self._len, 1), method='box-cox', standardize=standard)
        return _check_type(data=(i[0] for i in _check_list(data=arr)), value_type=self._value_type)

    def sklearn_yeo_johnson(self, standard: bool = True):
        """Postive values and negative values"""
        arr = power_transform(X=np.array(self._data).reshape(self._len, 1), method='yeo-johnson', standardize=standard)
        return _check_type(data=(i[0] for i in _check_list(data=arr)), value_type=self._value_type)

    def sklearn_quantile(self, n_quantiles: int = 25, output_distribution: str = 'uniform'):
        """Recommended to not do before splitting"""
        """Also accepts 'normal' """
        arr = quantile_transform(X=np.array(self._data).reshape(self._len, 1), n_quantiles=n_quantiles,
                                 output_distribution=output_distribution)
        return _check_type(data=(i[0] for i in _check_list(data=arr)), value_type=self._value_type)

    def sklearn_robust_scaling(self, with_centering: bool = True, with_scaling: bool = True, quantile_range: tuple = (25.0, 75.0)):
        """Recommended to not do before splitting"""
        arr = robust_scale(X=np.array(self._data).reshape(self._len, 1), with_centering=with_centering,
                           with_scaling=with_scaling, quantile_range=quantile_range)
        return _check_type(data=(i[0] for i in _check_list(data=arr)), value_type=self._value_type)


    def __repr__(self):
        return 'TransformData'

    @property
    def clean_data(self):
        return self._cleanData
