"""
Regression class object.

Usage:
 ./models/regression.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import Union, List
from dataclasses import dataclass
import numpy as np
from statsmodels import regression
from statsmodels.tools import add_constant
from pyjr.utils.cleandata import CleanData
from pyjr.utils.base import _check_list


@dataclass
class Regression:
    """

    Calculate a linear regression.

    :param doc_filter: Input DocumentFilter.
    :type doc_filter: DocumentFilter
    :param x_column: Name of column or columns to be used in regression analysis.
    :type x_column: str, or List[str]
    :param y_column: Name of column to be used as y variable in regression.
    :type y_column: str
    :example:
    :note: This will return a Regression object with regression result information.

    """
    x_data: CleanData
    y_data: CleanData

    def __init__(self, x_data, y_data):

        if x_data.len != y_data.len:
            raise AttributeError('X and Y data are not the same length.')

        x = add_constant(np.array(x_data.data, dtype=float))
        y = np.array(y_data.data)
        model = regression.linear_model.OLS(y, x).fit()

        self._constant_coef = model.params[0]
        self._item_coef = model.params[1]
        # self._coefficients = None
        # self._confidence_bounds = None
        self._lower_conf = model.conf_int()[1, :2][0]
        self._upper_conf = model.conf_int()[1, :2][1]
        self._pvalue = model.pvalues[1]
        self._r2 = model.rsquared
        self._resid = _check_list(data=model.resid)
        self._bse = model.bse
        self._mse = model.mse_model
        self._ssr = model.ssr
        self._ess = model.ess

    def __repr__(self):
        return 'Regression Analysis'

    @property
    def r2(self):
        """Returns R Squared"""
        return self._r2

    @property
    def constant_coefficient(self):
        """Returns Constant Coefficient, if only one x_column is provided"""
        return self._constant_coef

    @property
    def x_coefficient(self):
        """Returns X Coefficient, if only one x_column is provided"""
        return self._item_coef

    @property
    def lower_confidence(self):
        """Returns Lower Confidence Value, if only one x_column is provided"""
        return self._lower_conf

    @property
    def upper_confidence(self):
        """Returns Upper Confidence Value, if only one x_column is provided"""
        return self._upper_conf

    @property
    def pvalue(self):
        """Returns P Value or Values"""
        return self._pvalue

    @property
    def residuals(self):
        """Returns residuals"""
        return self._resid

    @property
    def mse(self):
        """Returns Mean Squared Error"""
        return self._mse

    @property
    def ssr(self):
        """Returns Sum of Squared Residuals"""
        return self._ssr

    @property
    def ess(self):
        """Returns Sum of Squared Error"""
        return self._ess

    # @property
    # def confidence(self):
    #     """Returns Confidence Values, if more than one x_column is provided"""
    #     return self._coefficients
    #
    # @property
    # def coefficients(self):
    #     """Returns Coefficient Values, if more than one x_column is provided"""
    #     return self._confidence_bounds