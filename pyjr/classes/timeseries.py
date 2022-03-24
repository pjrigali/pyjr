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
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from scipy import signal
from pyts.bag_of_words import BagOfWords, WordExtractor
from pyts.approximation import DiscreteFourierTransform, MultipleCoefficientBinning, PiecewiseAggregateApproximation, SymbolicAggregateApproximation, SymbolicFourierApproximation
from pyts.decomposition import SingularSpectrumAnalysis
from pyts.transformation import BOSS, ShapeletTransform
from pyts.metrics import boss, dtw
from scipy.stats import ks_2samp, ttest_ind


@dataclass
class TimeSeries:

    __slots__ = ["cleanData", "data", "name"]

    def __init__(self, data: Union[Data, PreProcess]):
        self.cleanData = data
        self.data = data.data
        self.name = "TS_" + data.name

    def __repr__(self):
        return "TimeSeriesData"

    def add_fft(self):
        y, Pxx = signal.periodogram(self.data, fs=self.cleanData.len, window='hanning', scaling='spectrum')
        Pxx = np.argsort(np.abs(Pxx))[::-1][1:10]
        self.data = tuple([1 / y[i] for i in Pxx])
        return self

    def add_zero_crossing(self, rate: int = 100):
        sig = self.data
        indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
        crossings = [i - sig[i] / (sig[i + 1] - sig[i]) for i in indices]
        self.data = np.mean(np.diff(crossings) / rate)
        return self

    def add_bag_of_words(self, window_size: Union[int, int] = 10, word_size: Union[int, float] = 4,
                         strategy: str = 'normal', n_bins: int = 4):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist.}
        bow = BagOfWords(window_size=window_size, word_size=word_size, strategy=strategy, n_bins=n_bins)
        self.data = bow.transform(self.data)
        self.name = "TS_BagOfWords"
        return self

    def add_word_extractor(self, window_size: Union[int, float] = 0.1, window_step: Union[int, float] = 1):
        word = WordExtractor(window_size=window_size, window_step=window_step)
        self.data = word.transform(self.data)
        self.name = "TS_WordExtractor"
        return self

    def add_DFT(self, n_coef: Union[int, float] = None):
        transfomer = DiscreteFourierTransform(n_coefs=n_coef)
        self.data = transfomer.fit_transform(self.data)
        self.name = "TS_DiscreteFouruerTransform"
        return self

    def add_MCB(self, n_bins: int = 4, strategy: str = 'quantile'):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist., 'entropy': Bin edges calc through information gain}
        transformer = MultipleCoefficientBinning(n_bins=n_bins, strategy=strategy)
        self.data = transformer.fit_transform(self.data)
        self.name = "TS_MultipleCoefficientBinning"
        return self

    def add_SAX(self, n_bins: int = 4, strategy: str = 'quantile'):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist.}
        transformer = SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)
        self.data = transformer.fit_transform(self.data)
        self.name = "TS_SymbolicAggregateApproximation"
        return self

    def add_PAA(self, window_size: Union[int, float] = 4):
        transformer = PiecewiseAggregateApproximation(window_size=window_size)
        self.data = transformer.fit_transform(self.data)
        self.name = "TS_PiecewiseAggregateApproximation"
        return self

    def add_SFA(self, n_bins: int = 4, strategy: str = 'quantile'):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist., 'entropy': Bin edges calc through information gain}
        transformer = SymbolicFourierApproximation(n_bins=n_bins, strategy=strategy)
        self.data = transformer.fit_transform(self.data)
        self.name = "TS_SymbolicFourierApproximation"
        return self

    def add_SSA(self, window_size: Union[int, float] = 4):
        transformer = SingularSpectrumAnalysis(window_size=window_size)
        self.data = transformer.fit_transform(self.data)
        self.name = "TS_SingularSpectrumAnalysis"
        return self

    def add_BOSS(self, window_size: Union[int, int] = 10, word_size: Union[int, float] = 4,
                 strategy: str = 'normal', n_bins: int = 4):
        # {'uniform': bins have identical widths, 'quantile': Same number of points., 'normal': Bin edges are from normal dist. 'entropy'}
        bow = BOSS(window_size=window_size, word_size=word_size, strategy=strategy, n_bins=n_bins)
        self.data = bow.fit_transform(self.data)
        self.name = "TS_BOSS"
        return self

    def add_Shapelet(self, data, criterion: str = "mutual_info"):
        # "anova"
        bow = ShapeletTransform(criterion=criterion)
        self.data = bow.fit_transform(self.data, data.data)
        self.name = "TS_ShapeletTransform"
        return self

    def get_boss(self, data):
        return boss(x=self.data, y=data.data)

    def get_dtw(self, data, dist: str = "square", method: str = "classic"):
        # {"square", "absolute", "precomputed", "callable"}
        # {"classic", "sakoechiba", "itakura", "region", "multiscale", "fast"}
        # more options if .show_options()
        return dtw(x=self.data, y=data.data, dist=dist, method=method)

    # Distance
    def get_eucludian_distance(self, other) -> float:
        return math.dist(self.data, other.data) / self.data.__len__()

    # MAPE
    # can not handle zero in denominator
    def get_mape(self, other) -> float:
        actual = np.array([i if i != 0.0 else .01 for i in self.data])
        pred = np.array([i if i != 0.0 else .01 for i in other.data])
        return np.mean(np.abs((actual - pred) / actual)) * 100

    # Auto-Correlation
    def get_auto_corr(self, other, num: int = 50) -> float:

        def acf(x, length):
            return [1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)]

        return np.corrcoef(acf(x=self.data, length=num), acf(x=other.data, length=num))[0, 1]

    # Correlation
    def get_corr(self, other) -> float:
        return np.corrcoef(self.data, other.data)[0, 1]

    # T Test to compare means
    def get_compare_means(self, other) -> float:
        return ttest_ind(a=self.data, b=other.data).pvalue

    # Kolmogorov-smirnov to see if from the same distribution
    def get_kol_smirnov(self, other) -> float:
        return ks_2samp(data1=self.data, data2=other.data).pvalue
