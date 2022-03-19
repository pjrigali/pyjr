from dataclasses import dataclass
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import random
from pyjr.utils.data import Data
from pyjr.utils.preprocess import PreProcess
from pyjr.utils.model_data import ModelingData
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class Classifier:

    __slots__ = ["data", "pred", "coef", "intercept", "scores"]

    def __init__(self, data: ModelingData):
        self.data = data
        self.pred = None
        self.coef = None
        self.intercept = None
        self.scores = None

    def __repr__(self):
        return "Classifier"

    def ridge(self):
        cls = RidgeClassifier().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def SVC(self, kernel: str = 'rbf', gamma: str = 'scale'):
        """Benefits from standardizing"""
        # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        # {'scale', 'auto'}
        cls = svm.SVC(kernel=kernel, gamma=gamma).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def NuSVC(self, kernel: str = 'rbf', gamma: str = 'scale'):
        """Benefits from standardizing"""
        # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
        # {'scale', 'auto'}
        cls = svm.NuSVC(kernel=kernel, gamma=gamma).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def svm(self):
        """Benefits from standardizing"""
        reg = svm.LinearSVC().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def SGDclass(self, loss: str = 'hinge'):
        # The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a
        # regression loss: ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        """Benefits from standardizing"""
        reg = SGDClassifier().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def knnClass(self, num: int = 5, weights: str = 'uniform'):
        reg = KNeighborsClassifier(n_neighbors=num, weights=weights).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def knnCentroid(self):
        reg = NearestCentroid().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def gaussian(self):
        # from sklearn.gaussian_process.kernels import RBF
        reg = GaussianProcessClassifier(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def tree(self):
        reg = DecisionTreeClassifier(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def ada(self):
        reg = AdaBoostClassifier(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def forest(self):
        reg = RandomForestClassifier(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def mlp(self, activation: str = 'relu', solver: str = 'adam', learning_rate: str = 'constant'):
        # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        # {‘lbfgs’, ‘sgd’, ‘adam’}
        reg = MLPClassifier(random_state=0, activation=activation, solver=solver, learning_rate=learning_rate)
        reg.fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercepts_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def nb(self):
        reg = GaussianNB().fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def log(self):
        reg = LogisticRegression(random_state=0).fit(X=self.data.x_train, y=self.data.y_train)
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.pred = reg.predict(X=self.data.x_test)
        return self

    def scores(self):
        self.scores = {"accuracy": accuracy_score(y_true=self.data.y_test, y_pred=self.pred),
                       "auc": auc(x=self.data.x_test, y=self.data.y_test),
                       "conf matrix": confusion_matrix(y_true=self.data.y_test, y_pred=self.pred),
                       "f1": f1_score(y_true=self.data.y_test, y_pred=self.pred),
                       "precision": precision_score(y_true=self.data.y_test, y_pred=self.pred),
                       "recall": recall_score(y_true=self.data.y_test, y_pred=self.pred)}
        return self
