from dataclasses import dataclass
from pyjr.utils.model_data import ModelingData
from pyjr.utils.base import _add_constant, _round_to
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
import statsmodels.api as sm


@dataclass
class FeaturePerf:

    __slots__ = ["modeling_data", "reg_results"]

    def __init__(self, data: ModelingData):
        self.modeling_data = data
        self.reg_results = None

    def __repr__(self):
        return "FeaturePerformance"

    def get_regression(self):
        """Return r2, pred to true correlation, and mean of residuals"""
        results = {}
        dic = dict(zip(range(len(self.modeling_data.x_data_names)), self.modeling_data.x_data_names))
        for i in range(len(self.modeling_data.x_data_names)):
            key = dic[i]
            results[key] = {}
            x = _add_constant(data=self.modeling_data.x_train[:, i])
            y = self.modeling_data.y_train
            lin_reg = sm.OLS(y, x).fit()
            pred = lin_reg.predict(_add_constant(data=self.modeling_data.x_test[:, i]))
            flat_ytest = self.modeling_data.y_test.reshape(1, pred.shape[0]).tolist()[0]
            results[key]["r2"] = _round_to(data=lin_reg.rsquared, val=100, remainder=True)
            results[key]['pred_true_coef'] = _round_to(data=np.corrcoef(pred, flat_ytest)[0, 1], val=100, remainder=True)
            results[key]['residuals_mean'] = _round_to(data=lin_reg.resid.mean(), val=100, remainder=True)
        self.reg_results = results
        return self






