"""
Classifiers class.

Usage:
 ./models/classification.py

Author:
 Peter Rigali - 2022-03-19
"""
from dataclasses import dataclass
from typing import Union
from pyjr.classes.model_data import ModelingData
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from pyts.classification import TSBF, SAXVSM, BOSSVS
from pyts.multivariate.classification import MultivariateClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from pyjr.utils._class_functions import _get_fit_pred


@dataclass
class Classifier:

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
        return "Classifier"

    def add_ridge(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = RidgeClassifier().fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_RidgeClassifier"
        return self

    def add_svc(self, kernel: str = 'rbf', gamma: str = 'scale', fit: str = 'train', pred: str = 'test'):
        """Benefits from standardizing"""
        # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        # {'scale', 'auto'}
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = svm.SVC(kernel=kernel, gamma=gamma).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_SVC"
        return self

    def add_nusvc(self, kernel: str = 'rbf', gamma: str = 'scale', fit: str = 'train', pred: str = 'test'):
        """Benefits from standardizing"""
        # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        # {'scale', 'auto'}
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = svm.NuSVC(kernel=kernel, gamma=gamma).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_NuSVC"
        return self

    def add_svm(self, fit: str = 'train', pred: str = 'test'):
        """Benefits from standardizing"""
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = svm.LinearSVC().fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_LinearSVC"
        return self

    def add_sgdclass(self, loss: str = 'hinge', fit: str = 'train', pred: str = 'test'):
        # The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a
        # regression loss: ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        """Benefits from standardizing"""
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = SGDClassifier(loss=loss).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_SGDClassifier"
        return self

    def add_knnclass(self, num: int = 5, weights: str = 'uniform', fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = KNeighborsClassifier(n_neighbors=num, weights=weights).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_KNeighborsClassifier"
        return self

    def add_knncentroid(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = NearestCentroid().fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_NearestCentroid"
        return self

    def add_gaussian(self, fit: str = 'train', pred: str = 'test'):
        # from sklearn.gaussian_process.kernels import RBF
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = GaussianProcessClassifier(random_state=0).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_GaussianProcessClassifier"
        return self

    def add_tree(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = DecisionTreeClassifier(random_state=0).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_DecisionTreeClassifier"
        return self

    def add_ada(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = AdaBoostClassifier(random_state=0).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_AdaBoostClassifier"
        return self

    def add_forest(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = RandomForestClassifier(random_state=0).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_RandomForestClassifier"
        return self

    def add_mlp(self, activation: str = 'relu', solver: str = 'adam', learning_rate: str = 'constant',
                fit: str = 'train', pred: str = 'test'):
        # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        # {‘lbfgs’, ‘sgd’, ‘adam’}
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = MLPClassifier(random_state=0, activation=activation, solver=solver, learning_rate=learning_rate)
        cls.fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercepts_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_MLPClassifier"
        return self

    def add_nb(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = GaussianNB().fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_GaussianNB"
        return self

    def add_log(self, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = LogisticRegression(random_state=0).fit(X=x, y=y)
        self.model = cls
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_LogisticRegression"
        return self

    # TimeSeries
    def add_tsbf(self, random_state: int = 0, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = TSBF(random_state=random_state).fit(X=x, y=y)
        self.model = cls
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_TSBF"
        return self

    def add_szxvsm(self, window_size: Union[int, float] = 0.5, word_size: Union[int, float] = 0.5, n_bins: int = 4,
                   stategy: str = "normal", fit: str = 'train', pred: str = 'test'):
        # {'uniform': bins have identical widths, 'quantile': Same number of points.,
        # 'normal': Bin edges are from normal dist.}
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = SAXVSM(window_size=window_size, word_size=word_size, n_bins=n_bins, strategy=stategy).fit(X=x, y=y)
        self.model = cls
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_SAXVSM"
        return self

    def add_bossvs(self, window_size: int = 4, word_size: int = 4, n_bins: int = 4, fit: str = 'train',
                   pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = BOSSVS(window_size=window_size, word_size=word_size, n_bins=n_bins).fit(X=x, y=y)
        self.model = cls
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_BOSSVS"
        return self

    def add_mc(self, estimator=BOSSVS(), fit: str = 'train', pred: str = 'test'):
        # {estimator object or list of}
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cls = MultivariateClassifier(estimator=estimator).fit(X=x, y=y)
        self.model = cls
        self.pred = cls.predict(X=_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.name = "CLS_MultivariateClassifier"
        return self

    # Metrics
    def add_scores(self, pred: str = 'test'):
        y_true = _get_fit_pred(data=self.data, fp='score', ttv=pred)
        self.scores = {"accuracy": accuracy_score(y_true=y_true, y_pred=self.pred),
                       "f1": f1_score(y_true=y_true, y_pred=self.pred),
                       "recall": recall_score(y_true=y_true, y_pred=self.pred),
                       "precision": precision_score(y_true=y_true, y_pred=self.pred),
                       "roc_auc": roc_auc_score(y_true=y_true, y_score=self.pred),
                       "conf_matrix": confusion_matrix(y_true=y_true, y_pred=self.pred)}
        return self
