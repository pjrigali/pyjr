"""
Time Series class.

Usage:
 ./utils/timeseries.py

Author:
 Peter Rigali - 2022-03-19
"""
from dataclasses import dataclass
from typing import Union
import numpy as np
import math
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from scipy import signal
from pyts.bag_of_words import BagOfWords, WordExtractor
from pyts.approximation import DiscreteFourierTransform, MultipleCoefficientBinning, PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation, SymbolicFourierApproximation
from pyts.decomposition import SingularSpectrumAnalysis
from pyts.transformation import BOSS, ShapeletTransform
from pyts.metrics import boss, dtw
from scipy.stats import ks_2samp, ttest_ind
from pyjr.utils.tools.math import _min
from pyjr.utils.tools.array import _ts_arr


@dataclass
class TimeSeries:

    __slots__ = ["cleanData", "data", "name", 'ts_data']

    def __init__(self, data: Union[Data, PreProcess]):
        self.cleanData = data
        self.data = None

        self.ts_data = None
        if isinstance(data.data, (tuple, list)):
            self.ts_data = _ts_arr(data=data.data)

        if isinstance(data.name, (list, tuple)):
            self.name = ['TS_' + name for name in data.name]
        else:
            self.name = "TS_" + data.name

    def __repr__(self):
        return "TimeSeriesData"

    def add_fft(self):
        y, Pxx = signal.periodogram(self.cleanData.data, fs=self.cleanData.len, window='hanning', scaling='spectrum')
        Pxx = np.argsort(np.abs(Pxx))[::-1][1:10]
        self.data = tuple([1 / y[i] for i in Pxx])
        return self

    def add_zero_crossing(self, rate: int = 100):
        sig = self.cleanData.array()
        indices = np.nonzero(np.ravel((sig[1:] >= 0) & (sig[:-1] < 0)))
        crossings = [i - sig[i] / (sig[i + 1] - sig[i]) for i in indices]
        self.data = np.mean(np.diff(crossings) / rate)
        return self

    def add_bag_of_words(self, window_size: Union[int, float] = 10, word_size: Union[int, float] = 4,
                         strategy: str = 'normal', n_bins: int = 4):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist.}
        # Requires 2d Array
        bow = BagOfWords(window_size=window_size, word_size=word_size, strategy=strategy, n_bins=n_bins)
        self.data = bow.transform(self.ts_data)
        self.name = "TS_BagOfWords"
        return self

    def add_DFT(self, n_coef: Union[int, float] = None):
        #requires 2d Array
        transfomer = DiscreteFourierTransform(n_coefs=n_coef)
        self.data = transfomer.fit_transform(self.ts_data)
        self.name = "TS_DiscreteFouruerTransform"
        return self

    def add_MCB(self, n_bins: int = 4, strategy: str = 'quantile'):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist., 'entropy': Bin edges calc through information gain}
        #requires [[1,1,1], [1,1,1], [1,1,1]]
        transformer = MultipleCoefficientBinning(n_bins=n_bins, strategy=strategy)
        self.data = transformer.fit_transform(self.ts_data)
        self.name = "TS_MultipleCoefficientBinning"
        return self

    def add_SAX(self, n_bins: int = 4, strategy: str = 'quantile'):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist.}
        #requires 2d array
        transformer = SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)
        self.data = transformer.fit_transform(self.ts_data)
        self.name = "TS_SymbolicAggregateApproximation"
        return self

    def add_word_extractor(self, window_size: Union[int, float] = 0.1, window_step: Union[int, float] = 1):
        # requires add_SAX before
        word = WordExtractor(window_size=window_size, window_step=window_step)
        self.data = word.transform(self.data)
        self.name = "TS_WordExtractor"
        return self

    def add_PAA(self, window_size: Union[int, float] = 4):
        # requires 2d array
        transformer = PiecewiseAggregateApproximation(window_size=window_size)
        self.data = transformer.fit_transform(self.ts_data)
        self.name = "TS_PiecewiseAggregateApproximation"
        return self

    # def add_SFA(self, n_bins: int = 4, strategy: str = 'quantile'):
    #     # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist., 'entropy': Bin edges calc through information gain}
    #     #requires 2d array
    #     transformer = SymbolicFourierApproximation(n_bins=n_bins, strategy=strategy)
    #     self.data = transformer.fit_transform(self.ts_data)
    #     self.name = "TS_SymbolicFourierApproximation"
    #     return self

    def add_SSA(self, window_size: Union[int, float] = 4):
        transformer = SingularSpectrumAnalysis(window_size=window_size)
        # requires 2d array
        self.data = transformer.fit_transform(self.ts_data)
        self.name = "TS_SingularSpectrumAnalysis"
        return self

    def add_BOSS(self, window_size: Union[int, int] = 10, word_size: Union[int, float] = 4,
                 strategy: str = 'normal', n_bins: int = 4):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist. 'entropy'}
        # requires 2d array
        bow = BOSS(window_size=window_size, word_size=word_size, strategy=strategy, n_bins=n_bins, sparse=False)
        self.data = bow.fit_transform(self.ts_data)
        self.name = "TS_BOSS"
        return self

    # Comparison
    # def add_Shapelet(self, data: Union[Data, PreProcess], criterion: str = "mutual_info"):
    #     # "anova"
    #     # X = [[0, 2, 3, 4, 3, 2, 1],
    #     #      [0, 1, 3, 4, 3, 4, 5],
    #     #      [2, 1, 0, 2, 1, 5, 4],
    #     #      [1, 2, 2, 1, 0, 3, 5]]
    #     # y = [0, 0, 1, 1]
    #     bow = ShapeletTransform(criterion=criterion)
    #     self.data = bow.fit_transform(self.cleanData.array(axis=1), data.data)
    #     self.name = "TS_ShapeletTransform"
    #     return self

    def get_boss(self, data: Union[Data, PreProcess]):
        return boss(x=self.cleanData.data, y=data.data)

    def get_dtw(self, data: Union[Data, PreProcess], dist: str = "square", method: str = "classic"):
        # {"square", "absolute", "precomputed", "callable"}
        # {"classic", "sakoechiba", "itakura", "region", "multiscale", "fast"}
        # more options if .show_options()
        return dtw(x=self.cleanData.data, y=data.data, dist=dist, method=method)

    # Distance
    def get_euclidean_distance(self, data: Union[list, tuple]) -> float:
        """Returns the distance between two lists or tuples."""
        if isinstance(self.cleanData.data, (list, tuple)) and isinstance(data, (list, tuple)):
            return math.dist(self.cleanData.data, data) / self.cleanData.len
        else:
            raise AttributeError("Both data and other data need to be a list or tuple.")

    # MAPE
    # can not handle zero in denominator
    def get_mape(self, data: list, min_value: float = 0.01) -> float:
        """Returns the MAPE between two lists or tuples."""
        if isinstance(self.cleanData.data, np.ndarray):
            raise AttributeError('Internal data needs to be in list format')
        elif isinstance(self.cleanData.data, tuple):
            self.cleanData.data = list(self.cleanData.data)
        if isinstance(data, np.ndarray):
            raise AttributeError('External data needs to be in list format')
        elif isinstance(data, tuple):
            data = list(data)
        if isinstance(self.cleanData.data, list) and isinstance(data, list):
            if _min(d=self.cleanData.data) == 0:
                actual = np.array([i if i != 0.0 else min_value for i in self.cleanData.data])
            else:
                actual = self.cleanData.array()
            if _min(d=data) == 0:
                pred = np.array([i if i != 0.0 else min_value for i in data])
            else:
                pred = np.array(data)
            return np.mean(np.abs((actual - pred) / actual)) * 100
        else:
            raise AttributeError("Both data and other data need to be a list or tuple.")

    # Auto-Correlation
    def get_auto_corr(self, data: Union[list, tuple], num: int = 50) -> float:
        """Returns the correlation of auto-corr between two lists or tuples."""
        if isinstance(self.cleanData.data, (list, tuple)) and isinstance(data, (list, tuple)):
            def acf(x, length):
                return [1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)]
            return np.corrcoef(acf(x=self.cleanData.data, length=num), acf(x=data, length=num))[0, 1]
        else:
            raise AttributeError("Both data and other data need to be a list or tuple.")

    # Correlation
    def get_corr(self, data: Union[list, tuple]) -> float:
        """Returns the correlation of two lists or tuples."""
        if isinstance(self.cleanData.data, (list, tuple)) and isinstance(data, (list, tuple)):
            return np.corrcoef(self.cleanData.data, data)[0, 1]
        else:
            raise AttributeError("Both data and other data need to be a list or tuple.")

    # T Test to compare means
    def get_compare_means(self, data: Union[list, tuple]) -> float:
        """Returns the t-test, comparing means of two lists or tuples."""
        if isinstance(self.cleanData.data, (list, tuple)) and isinstance(data, (list, tuple)):
            return ttest_ind(a=self.cleanData.data, b=data).pvalue
        else:
            raise AttributeError("Both data and other data need to be a list or tuple.")

    # Kolmogorov-smirnov to see if from the same distribution
    def get_kol_smirnov(self, data: Union[list, tuple]) -> float:
        """Returns if teo lists or tuples come from the same distribution, as a pvalue."""
        if isinstance(self.cleanData.data, (list, tuple)) and isinstance(data, (list, tuple)):
            return ks_2samp(data1=self.cleanData.data, data2=data).pvalue
        else:
            raise AttributeError("Both data and other data need to be a list or tuple.")
