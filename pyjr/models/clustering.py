from dataclasses import dataclass
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import random
from pyjr.utils.data import Data
from pyjr.utils.preprocess import PreProcess
from pyjr.utils.model_data import ModelingData
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, AgglomerativeClustering, Birch
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, v_measure_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, completeness_score


@dataclass
class Cluster:

    __slots__ = ["data", "labels", "pred", "centers", "scores"]

    def __init__(self, data: ModelingData):
        self.data = data
        self.labels = None
        self.pred = None
        self.centers = None
        self.scores = None

    def __repr__(self):
        return "Cluster"

    def kmeans(self, num: int = 8, random_state: int = 0):
        kmeans = KMeans(n_clusters=num, random_state=random_state).fit(self.data.x_train)
        self.labels = kmeans.labels_
        self.pred = kmeans.predict(self.data.x_test)
        self.centers = kmeans.cluster_centers_
        return self

    def affinity(self, random_state: int = 0):
        cluster = AffinityPropagation(random_state=random_state).fit(self.data.x_train)
        self.labels = cluster.labels_
        self.pred = cluster.predict(self.data.x_test)
        self.centers = cluster.cluster_centers_
        return self

    def mean_shift(self, bandwidth: int = None):
        cluster = MeanShift(bandwidth=bandwidth).fit(self.data.x_train)
        self.labels = cluster.labels_
        self.pred = cluster.predict(self.data.x_test)
        self.centers = cluster.cluster_centers_
        return self

    def agglomerative(self, num: int = 2):
        cluster = AgglomerativeClustering(n_clusters=num).fit(self.data.x_train)
        self.labels = cluster.labels_
        self.pred = cluster.predict(self.data.x_test)
        self.centers = cluster.cluster_centers_
        return self

    def birch(self, num: int = 2):
        cluster = Birch(n_clusters=num).fit(self.data.x_train)
        self.labels = cluster.labels_
        self.pred = cluster.predict(self.data.x_test)
        self.centers = cluster.subcluster_centers_
        return self

    def scores(self):
        self.scores = {"rand": rand_score(labels_true=self.data.y_test, labels_pred=self.pred),
                       "adj rand": adjusted_rand_score(labels_true=self.data.y_test, labels_pred=self.pred),
                       "mutual info": mutual_info_score(labels_true=self.data.y_test, labels_pred=self.pred),
                       "v measure": v_measure_score(labels_true=self.data.y_test, labels_pred=self.pred),
                       "silhouette": silhouette_score(X=self.data.x_test, labels=self.labels),
                       "cont matrix": contingency_matrix(labels_true=self.data.y_test, labels_pred=self.pred),
                       "completeness": completeness_score(labels_true=self.data.y_test, labels_pred=self.pred),
                       "homogenity": homogeneity_score(labels_true=self.data.y_test, labels_pred=self.pred)}
        return self
