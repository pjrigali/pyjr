from dataclasses import dataclass
from pyjr.utils.cleandata import CleanData
from pyjr.utils.base import _check_type, _mean, _percentile, _median, _std


@dataclass
class TransformData:

    def __init__(self, data: CleanData):
        self._cleanData = data
        self._value_type = self._cleanData.inputs['value_type']
        self._data = self._cleanData.data
        self._len = self._cleanData.len
        self._ddof = self._cleanData.inputs['ddof']

    def normalize(self) -> list:
        max_min_val = self._cleanData.max - self._cleanData.min
        if max_min_val == 0.0:
            max_min_val = 1.0
        return _check_type(data=[(val - self._cleanData.min) / max_min_val for val in self._data],
                           value_type=self._value_type)

    def standardize(self) -> list:
        return _check_type(data=[(item - self._cleanData.mean) / self._cleanData.std for item in self._data],
                           value_type=self._value_type)

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

    @property
    def clean_data(self):
        return self._cleanData
