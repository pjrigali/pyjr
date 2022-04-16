"""
Regressor class.

Usage:
 ./models/regression.py

Author:
 Peter Rigali - 2022-03-19
"""
from dataclasses import dataclass
from pyjr.classes.model_data import ModelingData
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, BayesianRidge, ARDRegression
from sklearn.linear_model import SGDRegressor, QuantileRegressor, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_log_error, median_absolute_error, r2_score, mean_absolute_percentage_error
from pyjr.utils._class_functions import _get_fit_pred


@dataclass
class Regressor:

    __slots__ = ["data", "pred", "coef", "intercept", "scores", "name", "model"]

    def __init__(self, data: ModelingData):
        self.data = data
        self.pred = None
        self.coef = None
        self.intercept = None
        self.scores = None
        self.name = None
        self.model = None

    def __repr__(self):
        return "Regressor"

    def add_linear(self, fit: str = 'train', pred: str = 'test'):
        """Add Linear Regression."""
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = LinearRegression().fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_LinearRegression"
        return self

    def add_ridge(self, alpha: float = 1.0, fit: str = 'train', pred: str = 'test'):
        """Add Ridge Regression."""
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = Ridge(alpha=alpha).fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_Ridge"
        return self

    def add_lasso(self, alpha: float = 1.0, fit: str = 'train', pred: str = 'test'):
        """Add Lasso Regression."""
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = Lasso(alpha=alpha).fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_Lasso"
        return self

    def add_lassolars(self, alpha: float = 1.0, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = LassoLars(alpha=alpha).fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_LassoLars"
        return self

    def add_elasticnet(self, alpha: float = 1.0, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = ElasticNet(alpha=alpha, random_state=0).fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_ElasticNet"
        return self

    def add_bayesridge(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = BayesianRidge().fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_BayesianRidge"
        return self

    def add_ard(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = ARDRegression().fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_ARDRegression"
        return self

    def add_sgdregress(self, fit: str = 'train', pred: str = 'test'):
        """Benefits from standardizing"""
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = SGDRegressor().fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_SGDRegressor"
        return self

    def add_quantile(self, q: float = 0.5, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = QuantileRegressor(quantile=q).fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_QuantileRegressor"
        return self

    def add_knn(self, num: int = 5, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = KNeighborsRegressor(n_neighbors=num).fit(X=x, y=y)
        self.model = reg
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_KNeighborsRegressor"
        return self

    def add_gaussian(self, fit: str = 'train', pred: str = 'test'):
        # from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
        # kernel = DotProduct() + WhiteKernel()
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = GaussianProcessRegressor(random_state=0).fit(X=x, y=y)
        self.model = reg
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_GaussianProcessRegressor"
        return self

    def add_tree(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = DecisionTreeRegressor(random_state=0).fit(X=x, y=y)
        self.model = reg
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_DecisionTreeRegressor"
        return self

    def add_ada(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = AdaBoostRegressor(random_state=0).fit(X=x, y=y)
        self.model = reg
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_AdaBoostRegressor"
        return self

    def add_forest(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = RandomForestRegressor(random_state=0).fit(X=x, y=y)
        self.model = reg
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_RandomForestRegressor"
        return self

    def add_mlp(self, activation: str = 'relu', solver: str = 'adam', learning_rate: str = 'constant',
                fit: str = 'train', pred: str = 'test'):
        # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        # {‘lbfgs’, ‘sgd’, ‘adam’}
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = MLPRegressor(random_state=0, activation=activation, solver=solver, learning_rate=learning_rate)
        reg.fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coefs_
        self.intercept = reg.intercepts_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_MLPRegressor"
        return self

    def add_huber(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        reg = HuberRegressor().fit(X=x, y=y)
        self.model = reg
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "REG_HuberRegressor"
        return self

    def add_scores(self, pred: str = 'test'):
        y_true = _get_fit_pred(data=self.data, fp='score', ttv=pred)
        # root mean square error prefered
        self.scores = {"explianed variance": explained_variance_score(y_true=y_true, y_pred=self.pred),
                       "max error": max_error(y_true=y_true, y_pred=self.pred),
                       "mean abs error": mean_absolute_error(y_true=y_true, y_pred=self.pred),
                       "mean squared error": mean_squared_error(y_true=y_true, y_pred=self.pred),
                       "mean squared log error": mean_squared_log_error(y_true=y_true, y_pred=self.pred),
                       "median abs error": median_absolute_error(y_true=y_true, y_pred=self.pred),
                       "r2": r2_score(y_true=y_true, y_pred=self.pred),
                       "mean abs per error": mean_absolute_percentage_error(y_true=y_true, y_pred=self.pred)}
        return self
