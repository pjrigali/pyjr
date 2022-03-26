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


@dataclass
class Classifier:

    __slots__ = ["data", "pred", "coef", "intercept", "scores", "name"]

    def __init__(self, data: ModelingData):
        self.data = data
        self.pred = None
        self.coef = None
        self.intercept = None
        self.scores = None
        self.name = None

    def __repr__(self):
        return "Classifier"

    def add_ridge(self):
        cls = RidgeClassifier().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=self.data.x_test)
        self.name = "CLS_RidgeClassifier"
        return self

    def add_svc(self, kernel: str = 'rbf', gamma: str = 'scale'):
        """Benefits from standardizing"""
        # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        # {'scale', 'auto'}
        cls = svm.SVC(kernel=kernel, gamma=gamma).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=self.data.x_test)
        self.name = "CLS_SVC"
        return self

    def add_nusvc(self, kernel: str = 'rbf', gamma: str = 'scale'):
        """Benefits from standardizing"""
        # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        # {'scale', 'auto'}
        cls = svm.NuSVC(kernel=kernel, gamma=gamma).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = cls.coef_
        self.intercept = cls.intercept_
        self.pred = cls.predict(X=self.data.x_test)
        self.name = "CLS_NuSVC"
        return self

    def add_svm(self):
        """Benefits from standardizing"""
        reg = svm.LinearSVC().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_LinearSVC"
        return self

    def add_sgdclass(self, loss: str = 'hinge'):
        # The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a
        # regression loss: ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        """Benefits from standardizing"""
        reg = SGDClassifier(loss=loss).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_SGDClassifier"
        return self

    def add_knnclass(self, num: int = 5, weights: str = 'uniform'):
        reg = KNeighborsClassifier(n_neighbors=num, weights=weights).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_KNeighborsClassifier"
        return self

    def add_knncentroid(self):
        reg = NearestCentroid().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_NearestCentroid"
        return self

    def add_gaussian(self):
        # from sklearn.gaussian_process.kernels import RBF
        reg = GaussianProcessClassifier(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_GaussianProcessClassifier"
        return self

    def add_tree(self):
        reg = DecisionTreeClassifier(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_DecisionTreeClassifier"
        return self

    def add_ada(self):
        reg = AdaBoostClassifier(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_AdaBoostClassifier"
        return self

    def add_forest(self):
        reg = RandomForestClassifier(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_RandomForestClassifier"
        return self

    def add_mlp(self, activation: str = 'relu', solver: str = 'adam', learning_rate: str = 'constant'):
        # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        # {‘lbfgs’, ‘sgd’, ‘adam’}
        reg = MLPClassifier(random_state=0, activation=activation, solver=solver, learning_rate=learning_rate)
        reg.fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercepts_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_MLPClassifier"
        return self

    def add_nb(self):
        reg = GaussianNB().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_GaussianNB"
        return self

    def add_log(self):
        reg = LogisticRegression(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        self.name = "CLS_LogisticRegression"
        return self

    # TimeSeries
    def add_tsbf(self, random_state: int = 0):
        clf = TSBF(random_state=random_state).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = clf.feature_importance_
        self.pred = clf.pedict(X=self.data.x_test)
        self.intercept = None
        self.name = "CLS_TSBF"
        return self

    def add_szxvsm(self, window_size: Union[int, float] = 0.5, word_size: Union[int, float] = 0.5, n_bins: int = 4,
                   stategy: str = "normal"):
        # {'uniform': bins have identical widths, 'quantile': Same number of points.,
        # 'normal': Bin edges are from normal dist.}
        clf = SAXVSM(window_size=window_size, word_size=word_size, n_bins=n_bins,
                     strategy=stategy).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = None
        self.pred = clf.pedict(X=self.data.x_test)
        self.intercept = None
        self.name = "CLS_SAXVSM"
        return self

    def add_bossvs(self, window_size: int = 4, word_size: int = 4, n_bins: int = 4):
        clf = BOSSVS(window_size=window_size, word_size=word_size, n_bins=n_bins).fit(X=self.data.x_train,
                                                                                      y=self.data.y_train)
        self.coef = None
        self.pred = clf.pedict(X=self.data.x_test)
        self.intercept = None
        self.name = "CLS_BOSSVS"
        return self

    def add_mc(self, estimator=BOSSVS()):
        # {estimator object or list of}
        clf = MultivariateClassifier(estimator=estimator).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = None
        self.pred = clf.pedict(X=self.data.x_test)
        self.intercept = None
        self.name = "CLS_MultivariateClassifier"
        return self

    # Metrics
    def add_scores(self):
        self.scores = {"accuracy": accuracy_score(y_true=self.data.y_test, y_pred=self.pred),
                       "f1": f1_score(y_true=self.data.y_test, y_pred=self.pred),
                       "recall": recall_score(y_true=self.data.y_test, y_pred=self.pred),
                       "precision": precision_score(y_true=self.data.y_test, y_pred=self.pred),
                       "roc_auc": roc_auc_score(y_true=self.data.y_test, y_score=self.pred),
                       "conf_matrix": confusion_matrix(y_true=self.data.y_test, y_pred=self.pred)}
        return self
