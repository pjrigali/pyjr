"""
Cluster class.

Usage:
 ./models/cluster.py

Author:
 Peter Rigali - 2022-03-19
"""
from dataclasses import dataclass
from pyjr.classes.model_data import ModelingData
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, AgglomerativeClustering, Birch
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, v_measure_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, completeness_score
from pyjr.utils._class_functions import _get_fit_pred


@dataclass
class Cluster:

    __slots__ = ["data", "labels", "pred", "centers", "scores", "name", "model"]

    def __init__(self, data: ModelingData):
        self.data = data
        self.labels = None
        self.pred = None
        self.centers = None
        self.scores = None
        self.name = None
        self.model = None

    def __repr__(self):
        return "Cluster"

    def add_kmeans(self, num: int = 8, random_state: int = 0, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cluster = KMeans(n_clusters=num, random_state=random_state).fit(x)
        self.model = cluster
        self.labels = cluster.labels_
        self.pred = cluster.predict(_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.centers = cluster.cluster_centers_
        self.name = "CLU_KMeans"
        return self

    def add_affinity(self, random_state: int = 0, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cluster = AffinityPropagation(random_state=random_state).fit(x)
        self.model = cluster
        self.labels = cluster.labels_
        self.pred = cluster.predict(_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.centers = cluster.cluster_centers_
        self.name = "CLU_AffinityPropagation"
        return self

    def add_mean_shift(self, bandwidth: int = None, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cluster = MeanShift(bandwidth=bandwidth).fit(x)
        self.model = cluster
        self.labels = cluster.labels_
        self.pred = cluster.predict(_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.centers = cluster.cluster_centers_
        self.name = "CLU_MeanShift"
        return self

    def add_agglomerative(self, num: int = 2, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cluster = AgglomerativeClustering(n_clusters=num).fit(x)
        self.model = cluster
        self.labels = cluster.labels_
        self.pred = cluster.predict(_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.centers = cluster.cluster_centers_
        self.name = "CLU_AgglomerativeClustering"
        return self

    def add_birch(self, num: int = 2, fit: str = 'train', pred: str = 'test'):
        x, y = _get_fit_pred(data=self.data, fp='fit', ttv=fit)
        cluster = Birch(n_clusters=num).fit(x)
        self.model = cluster
        self.labels = cluster.labels_
        self.pred = cluster.predict(_get_fit_pred(data=self.data, fp='pred', ttv=pred))
        self.centers = cluster.subcluster_centers_
        self.name = "CLU_Birch"
        return self

    def add_scores(self, pred: str = 'test'):
        y_true = _get_fit_pred(data=self.data, fp='score', ttv=pred)
        self.scores = {"rand": rand_score(labels_true=y_true, labels_pred=self.pred),
                       "adj rand": adjusted_rand_score(labels_true=y_true, labels_pred=self.pred),
                       "mutual info": mutual_info_score(labels_true=y_true, labels_pred=self.pred),
                       "v measure": v_measure_score(labels_true=y_true, labels_pred=self.pred),
                       "silhouette": silhouette_score(X=self.data.x_test, labels=self.labels),
                       "cont matrix": contingency_matrix(labels_true=y_true, labels_pred=self.pred),
                       "completeness": completeness_score(labels_true=y_true, labels_pred=self.pred),
                       "homogenity": homogeneity_score(labels_true=y_true, labels_pred=self.pred)}
        return self
