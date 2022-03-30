"""
Main file.

Usage:
 None

Author:
 Peter Rigali - 2022-03-10
"""
import pandas as pd
import numpy as np
import random
pd.set_option('display.max_columns', None)
pd.set_option('use_inf_as_na', True)
import pyjr.utils.simple as jr
from pyjr.plot.line import Line
from pyjr.plot.scatter import Scatter
from pyjr.plot.histogram import Histogram
from pyjr.plot.table import Table
from pyjr.plot.barchart import Bar
import matplotlib.pyplot as plt
from pyjr.classes.stat import Stat
if __name__ == '__main__':

    data = [random.randrange(1, 100, 1) for i in range(1000)]
    data1 = [random.randrange(1, 100, 1) for i in range(1000)]
    data2 = [random.randrange(0, 2, 1) for i in range(1000)]
    data3 = [str(random.randrange(0, 5, 1)) for i in range(1000)]
    from pyjr.utils.simple import oneHotEncode
    ohe = oneHotEncode(data3)

    from pyjr.classes.data import Data
    t = Data(data=data, name='Test Data')
    tt = Data(data=data1, name='Test Data1')
    ttt = Data(data=data2, name='Test Data2')
    ohe = Data(data=data3, name='Test Data3')

    from pyjr.classes.preprocess_data import PreProcess
    tttt = PreProcess(data=t).add_standardize(stat="mean")
    ttttt = PreProcess(data=t).add_normalize(stat="mean")

    # Line(data=[t, tttt, ttttt], limit=(0, 100))
    # plt.show()
    #
    # Scatter(data=[t, tttt, ttttt], limit=(0, 100), compare_two=True)
    # plt.show()
    #
    # Histogram(data=[t, tttt, ttttt], limit=(0, 100), include_norm='Test Data')
    # plt.show()
    #
    # Table(data=[t, tttt, ttttt], limit=(0, 50))
    # plt.show()

    # Bar(data=[t, tttt, ttttt], limit=(0, 50), vert_hor=False, include_mu=True)
    # plt.show()

    # from pyjr.classes.model_data import ModelingData
    # model_data = ModelingData()
    # model_data.add_multiple_xdata(data=(t, tt, tttt)).add_ydata(data=ttt).add_ohe_data(data=ohe)
    # explained = model_data.add_pca()

    # x_train, x_test, x_valid, y_train, y_test, y_valid = model_data.add_train_test_validate(train=0.70, test=0.20, valid=0.10)
    # y_balance = model_data.get_balance()

    # from pyjr.models.simple_regression import Regression
    # reg = Regression(x_data=t, y_data=tt)

    # from pyjr.classes.features import FeaturePerformance
    # feature_data = FeaturePerformance(data=model_data).add_regression()
    # feature_data = FeaturePerformance(data=model_data).add_outlier_std().add_outlier_var().add_outlier_regression().add_outlier_hist().add_outlier_knn().add_outlier_cooks_distance()
    # out = FeaturePerformance(data=model_data).add_outlier_std().get_outliers()


    # from pyjr.classes.timeseries import TimeSeries
    # ts = TimeSeries(data=t).get_dtw(data=tt)

    # from pyjr.classes.tuning import Tune
    # from sklearn.linear_model import SGDClassifier
    # x, y = model_data.x_train, model_data.y_train.tolist()
    # model = SGDClassifier()
    # tune = Tune(model=model, data=model_data)

    # @track
    # def test():
    #     for i in range(100):
    #         sum([random.randrange(1, 100, 1) for i in range(1000)])
    #     return
    # u = test()

    # from pyjr.utils.simple import outlier_cooks_distance, outlier_knn, outlier_hist, outlier_distance, outlier_regression, outlier_var, outlier_std
    # c = outlier_cooks_distance(x_data=t, y_data=tt)
    # knn = outlier_knn(x_data=t, y_data=tt)
    # h = outlier_hist(data=t)
    # d = outlier_distance(x_data=t, y_data=tt)
    # rr = outlier_regression(x_data=t, y_data=tt)
    # v = outlier_var(data=t)
    # st = outlier_std(data=t)

    # import inspect
    #
    # sig = inspect.signature(PreProcess)
    # sig1 = inspect.getmembers(PreProcess)
    # sig3 = inspect.getfullargspec(PreProcess)
    #
    t

