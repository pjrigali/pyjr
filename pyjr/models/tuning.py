"""
Hyperparameter tuning class.

Usage:
 ./models/tuning.py

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
from sklearn.model_selection import GridSearchCV


@dataclass
class Tune:

    __slots__ = ["data", "pred", "coef", "intercept", "scores"]

    def __init__(self, data: ModelingData):
        self.data = data
        self.pred = None
        self.coef = None
        self.intercept = None
        self.scores = None

    def __repr__(self):
        return "ModelTuning"
