"""
Hyper-parameter tuning class.

Usage:
 ./models/tuning.py

Author:
 Peter Rigali - 2022-03-19
"""
from dataclasses import dataclass
import numpy as np
from pyjr.classes.model_data import ModelingData
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve, permutation_test_score, validation_curve
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Tune:

    __slots__ = ["data", "pred", "model", "cv", "scores", "jobs", "train_sizes"]

    def __init__(self, model, data: ModelingData, jobs: int = 1, cv: int = 10):
        self.data = data
        self.model = model
        self.cv = cv
        self.scores = {}
        self.jobs = jobs

        self.scores['standard'] = tuple(cross_val_score(estimator=self.model, X=self.data.x_train, y=self.data.y_train, cv=self.cv, n_jobs=self.jobs).tolist())
        self.pred = tuple(cross_val_predict(estimator=self.model, X=self.data.x_train, y=self.data.y_train, cv=self.cv, n_jobs=self.jobs).tolist())
        lr = learning_curve(estimator=self.model, X=self.data.x_train, y=self.data.y_train, cv=self.cv, n_jobs=self.jobs)
        self.train_sizes, self.scores['training'], self.scores['testing'] = [tuple(i) for i in lr]
        self.scores['permutation'] = permutation_test_score(estimator=self.model, X=self.data.x_train, y=self.data.y_train, cv=self.cv, n_jobs=self.jobs)

        for key in ['testing', "training"]:
            temp = []
            for val in self.scores[key]:
                if isinstance(val, np.ndarray):
                    temp.append(tuple(val.tolist()))
                else:
                    temp.append(val)
            self.scores[key] = tuple(temp)

    def __repr__(self):
        return "ModelTuning"

    def add_score(self):
        self.scores['standard'] = tuple(cross_val_score(estimator=self.model, X=self.data.x_train, y=self.data.y_train, cv=self.cv, n_jobs=self.jobs).tolist())
        return self

    def add_pred(self):
        self.pred = tuple(cross_val_predict(estimator=self.model, X=self.data.x_train, y=self.data.y_train, cv=self.cv, n_jobs=self.jobs).tolist())
        return self

    def add_learning_curve(self):
        lr = learning_curve(estimator=self.model, X=self.data.x_train, y=self.data.y_train, cv=self.cv, n_jobs=self.jobs)
        self.train_sizes, self.scores['traing'], self.scores['testing'] = [tuple(i) for i in lr]
        return self

    def add_permutation_score(self):
        self.scores['permutation'] = permutation_test_score(estimator=self.model, X=self.data.x_train, y=self.data.y_train, cv=self.cv, n_jobs=self.jobs)
        return self
