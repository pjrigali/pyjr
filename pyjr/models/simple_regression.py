"""
Regression class object.

Usage:
 ./classes/simple_regression.py

Author:
 Peter Rigali - 2022-03-19
"""
from dataclasses import dataclass
from typing import Union
import numpy as np
from statsmodels import regression
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils._tools.general import _add_constant


@dataclass
class Regression:
    """

    Calculate linear regression.

    :param x_data: Input x Data.
    :type x_data: Data or PreProcess
    :param y_data: Input y Data.
    :type y_data: Data or PreProcess
    :example:
    :note: This will return a Regression object with regression result information.

    """
    __slots__ = "results"

    def __init__(self, x_data: Union[Data, PreProcess], y_data: Union[Data, PreProcess]):
        if x_data.len != y_data.len:
            raise AttributeError('X and Y data are not the same length.')
        model = regression.linear_model.OLS(y_data.array(), _add_constant(x_data.data)).fit()
        self.results = {att: getattr(model, att) for att in dir(model) if '__' not in att and att[0] != '_' and isinstance(getattr(model, att), (np.float_, np.ndarray, np.int_, np.str_, np.bool_))}

    def __repr__(self):
        return 'RegressionAnalysis'
