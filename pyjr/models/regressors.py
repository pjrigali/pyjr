"""
Regressor class.

Usage:
 ./models/regressors.py

Author:
 Peter Rigali - 2022-03-19
"""
from dataclasses import dataclass
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import random
from pyjr.utils.data import Data
from pyjr.utils.preprocess import PreProcess
from pyjr.utils.model_data import ModelingData
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, BayesianRidge, ARDRegression
from sklearn.linear_model import SGDRegressor, QuantileRegressor, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_log_error, median_absolute_error, r2_score, mean_absolute_percentage_error


@dataclass
class Regressor:

    __slots__ = ["data", "pred", "coef", "intercept", "scores"]

    def __init__(self, data: ModelingData):
        self.data = data
        self.pred = None
        self.coef = None
        self.intercept = None
        self.scores = None

    def __repr__(self):
        return "Regressor"

    def linear(self):
        reg = LinearRegression().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def ridge(self, alpha: float = 1.0):
        reg = Ridge(alpha=alpha).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def lasso(self, alpha: float = 1.0):
        reg = Lasso(alpha=alpha).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def lassoLars(self, alpha: float = 1.0):
        reg = LassoLars(alpha=alpha).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def elasticNet(self, alpha: float = 1.0):
        reg = ElasticNet(alpha=alpha, random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def bayesRidge(self):
        reg = BayesianRidge().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def ARD(self):
        reg = ARDRegression().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def SGDregress(self):
        """Benefits from standardizing"""
        reg = SGDRegressor().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def quantile(self, q: float = 0.5):
        reg = QuantileRegressor(quantile=q).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def knn(self, num: int = 5):
        reg = KNeighborsRegressor(n_neighbors=num).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def gaussian(self):
        # from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
        # kernel = DotProduct() + WhiteKernel()
        reg = GaussianProcessRegressor(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def tree(self):
        reg = DecisionTreeRegressor(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def ada(self):
        reg = AdaBoostRegressor(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def forest(self):
        reg = RandomForestRegressor(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def mlp(self, activation: str = 'relu', solver: str = 'adam', learning_rate: str = 'constant'):
        # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        # {‘lbfgs’, ‘sgd’, ‘adam’}
        reg = MLPRegressor(random_state=0, activation=activation, solver=solver, learning_rate=learning_rate)
        reg.fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercepts_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def huber(self):
        reg = HuberRegressor().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def scores(self):
        # root mean square error prefered
        self.scores = {"explianed variance": explained_variance_score(y_true=self.data.y_test, y_pred=self.pred),
                       "max error": max_error(y_true=self.data.y_test, y_pred=self.pred),
                       "mean abs error": mean_absolute_error(y_true=self.data.y_test, y_pred=self.pred),
                       "mean squared error": mean_squared_error(y_true=self.data.y_test, y_pred=self.pred),
                       "mean squared log error": mean_squared_log_error(y_true=self.data.y_test, y_pred=self.pred),
                       "median abs error": median_absolute_error(y_true=self.data.y_test, y_pred=self.pred),
                       "r2": r2_score(y_true=self.data.y_test, y_pred=self.pred),
                       "mean abs per error": mean_absolute_percentage_error(y_true=self.data.y_test, y_pred=self.pred)}
        return self
