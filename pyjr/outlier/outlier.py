"""
Various outlier detection functions.

Usage:
 ./outlier/outlier.py

Author:
 Peter Rigali - 2021-08-30
"""
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pyjr.utils.base import _mean, _median, _std, _percentile, _variance, _check_list, _sum
from pyjr.utils.cleandata import CleanData
per_dic = {-3: 0.001, -2: 0.023, -1: 0.159, 0: 0.50, 1: 0.841, 2: 0.977, 3: 0.999}


def stack(x_arr: np.ndarray, y_arr: np.ndarray, multi: Optional[bool] = False) -> np.ndarray:
    """

    Stacks x_arr and y_arr.

    :param x_arr: An array to stack.
    :type x_arr: np.ndarray
    :param y_arr: An array to stack.
    :type y_arr: np.ndarray
    :param mutli: If True, will stack based on multiple x_arr columns, default is False. *Optional*
    :type multi: bool
    :return: Array with a x column and a y column
    :rtype: np.ndarray
    :example: *None*
    :note: *None*

    """
    lst = []
    if multi:
        for i in range((x_arr.shape[1])):
            lst.append(np.vstack([x_arr[:, i].ravel(), y_arr[:, i].ravel()]).T)
        return np.array(lst)
    else:
        lst = np.vstack([x_arr.ravel(), y_arr.ravel()]).T
    return np.where(np.isnan(lst), 0, lst)


def _cent(x_lst: List[float], y_lst: List[float]) -> List[float]:
    """

    Calculate Centroid from x and y value(s).

    :param x_lst: A list of values.
    :type x_lst: List[float]
    :param y_lst: A list of values.
    :type y_lst: List[float]
    :returns: A list of x and y values representing the centriod of two lists.
    :rtype: List[float]
    :example: *None*
    :note: *None*

    """
    return [np.sum(x_lst) / len(x_lst), np.sum(y_lst) / len(y_lst)]


def _dis(cent1: List[float], cent2: List[float]) -> float:
    """

    Calculate Distance between two centroids.

    :param cent1: An x, y coordinate representing a centroid.
    :type cent1: List[float]
    :param cent2: An x, y coordinate representing a centroid.
    :type y_lst: List[float]
    :returns: A distance measurement.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    return round(np.sqrt((cent1[0] - cent2[0]) ** 2 + (cent1[1] - cent2[1]) ** 2), 4)


def outlier_std(data: CleanData, plus: bool = True, std_value: int = 2, return_ind: bool = False) -> np.ndarray:
    """

    Calculate Outliers using a simple std value.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param y_column: A target column. *Optional*
    :type y_column: str
    :param _std: A std threshold, default is 3. *Optional*
    :type _std: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    new_data = np.array(data.data)
    if data.min >= 0:
        if plus:
            ind = np.where(new_data <= _percentile(data=data.data, q=per_dic[std_value], length=data.len))[0]
        else:
            ind = np.where(new_data >= _percentile(data=data.data, q=per_dic[-std_value], length=data.len))[0]
    else:
        if plus:
            ind = np.where(new_data <= data.mean + data.std * std_value)[0]
        else:
            ind = np.where(new_data >= data.mean - data.std * std_value)[0]

    if return_ind:
        return ind
    else:
        return new_data[ind]


def outlier_var(data: CleanData, plus: Optional[bool] = True, std_value: int = 2,
                return_ind: bool = False) -> np.ndarray:
    """

    Calculate Outliers using a simple var value.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param y_column: A target column. *Optional*
    :type y_column: str
    :param per: A percent threshold, default is 0.95. *Optional*
    :type per: float
    :param plus: If True, will grab all values above the threshold. *Optional*
    :type plus: bool, default is True
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    lst = data.data.copy()
    temp_var = _variance(data=lst, ddof=data.inputs['ddof'])
    dev_based = np.array([temp_var - _variance(np.delete(lst, i), ddof=data.inputs['ddof']) for i, j in enumerate(lst)])

    if plus:
        q = _percentile(data=lst, length=data.len, q=per_dic[std_value])
        ind = np.where(dev_based <= q)[0]
    else:
        q = _percentile(data=lst, length=data.len, q=per_dic[-std_value])
        ind = np.where(dev_based >= q)[0]

    if return_ind:
        return ind
    else:
        return np.array(lst)[ind]


def outlier_regression(x_data: CleanData, y_data: CleanData, plus: Optional[bool] = True, std_value: Optional[int] = 2,
                       return_ind: bool = False) -> np.ndarray:
    """

    Calculate Outliers using regression.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param y_column: A column for y variables. *Optional*
    :type y_column: str
    :param std_value: A std threshold, default is 3. *Optional*
    :type std_value: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    arr = stack(np.array(x_data.data), np.array(y_data.data), False)
    ran = np.array(range(x_data.len))
    mu_y = np.zeros(len(arr) - 1)
    line_ys = []
    for i, j in enumerate(arr):
        xx, yy = np.delete(arr[:, 0], i), np.delete(arr[:, 1], i)
        w1 = (np.cov(xx, yy, ddof=1) / _variance(xx, ddof=1))[0, 1]
        new_y = w1 * ran[:-1] + (-1 * _mean(xx) * w1 + _mean(yy))
        mu_y = (mu_y + new_y) / 2
        line_ys.append(new_y)

    reg_based = np.array([np.mean(np.square(mu_y - j)) for i, j in enumerate(line_ys)])
    if plus:
        threshold = _percentile(data=reg_based, length=len(reg_based), q=per_dic[std_value])
        ind = np.where(reg_based <= threshold)[0]
    else:
        threshold = _percentile(data=reg_based, length=len(reg_based), q=per_dic[-std_value])
        ind = np.where(reg_based >= threshold)[0]

    if return_ind:
        return ind
    else:
        return arr[ind]


def outlier_distance(x_data: CleanData, y_data: CleanData, plus: Optional[bool] = True, std_value: int = 2,
                     return_ind: bool = False) -> np.ndarray:
    """

    Calculate Outliers using distance measurements.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param: data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param y_column: A column for y variables. *Optional*
    :type y_column: str
    :param std_value: A std threshold, default is 3. *Optional*
    :type std_value: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    arr = stack(np.array(x_data.data), np.array(y_data.data), False)
    cent_other = _cent(arr[:, 0], arr[:, 1])
    ran = range(0, x_data.len)
    x_y_other_centers = np.array([_dis(_cent(x_lst=[arr[i][0]], y_lst=[arr[i][1]]), cent_other) for i in ran])

    if plus:
        x_y_other_centers_std = _percentile(data=x_y_other_centers, length=len(x_y_other_centers), q=per_dic[std_value])
        ind = np.where(x_y_other_centers <= x_y_other_centers_std)[0]
    else:
        x_y_other_centers_std = _percentile(data=x_y_other_centers, length=len(x_y_other_centers), q=per_dic[-std_value])
        ind = np.where(x_y_other_centers >= x_y_other_centers_std)[0]

    if return_ind:
        return ind
    else:
        return arr[ind]


def outlier_hist(data: CleanData, plus: Optional[bool] = True, std_value: int = 2, return_ind: bool = False) -> np.ndarray:
    """

    Calculate Outliers using Histogram.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param: data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param per: A std threshold, default is 3. *Optional*
    :type per: float
    :param plus: If True, will grab all values above the threshold, default is 0.75. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    arr = np.array(data.data)
    n, b = np.histogram(arr, bins='sturges')

    if plus:
        qn = _percentile(data=data.data, q=per_dic[std_value], length=data.len)
        ind = np.where(n <= qn)[0]
        bin_edges = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind]
    else:
        qn = _percentile(data=data.data, q=per_dic[-std_value], length=data.len)
        ind = np.where(n >= qn)[0]
        bin_edges = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind]

    z_selected_ind = []
    for i, j in enumerate(arr):
        for k, l in bin_edges:
            if k >= j <= l:
                z_selected_ind.append(i)
                break

    # select = np.in1d(arr, arr[z_selected_ind])
    # return np.array([np.where(arr == i)[0][0] for i in arr[np.in1d(arr, arr[~select])]])
    if return_ind:
        return z_selected_ind
    else:
        return arr[z_selected_ind]


def outlier_knn(x_data: CleanData, y_data: CleanData, plus: Optional[bool] = True, std_value: int = 2,
                return_ind: bool = False) -> np.ndarray:
    """

    Calculate Outliers using KNN.

    :param arr: An Array to get data from. *Optional*
    :type arr: np.ndarray
    :param: data: A DataFrame to get data from. *Optional*
    :type data: pd.DataFrame
    :param x_column: A column for x variables. *Optional*
    :type x_column: str
    :param y_column: A column for y variables. *Optional*
    :type y_column: str
    :param std_value: A std threshold, default is 3. *Optional*
    :type std_value: int
    :param plus: If True, will grab all values above the threshold, default is True. *Optional*
    :type plus: bool
    :return: An array of indexes.
    :rtype: np.ndarray
    :example: *None*
    :note: If **arr** not passed, data and respective column names are required.

    """
    arr = stack(np.array(x_data.data), np.array(y_data.data), False)
    ran = range(0, x_data.len)
    test_centers = (_cent([arr[ind, 0]], [arr[ind, 1]]) for ind in ran)
    distances = [_dis(cent1=i, cent2=j) for i in test_centers for j in test_centers]

    if plus:
        threshold = _percentile(data=distances, q=per_dic[std_value], length=x_data.len)
        count_dic = {}
        for i, j in enumerate(arr):
            temp = arr[i, :] <= threshold
            count_dic[i] = _sum([1 for i in temp if i == True])
    else:
        threshold = _percentile(data=distances, q=per_dic[-std_value], length=x_data.len)
        count_dic = {}
        for i, j in enumerate(arr):
            temp = arr[i, :] >= threshold
            count_dic[i] = _sum([1 for i in temp if i == True])

    lst = []
    for val in _check_list(data=count_dic.values()):
        if isinstance(val, list):
            for val1 in val:
                lst.append(val1)
        else:
            lst.append(val)

    if plus:
        val1 = _percentile(data=lst, q=per_dic[std_value], length=x_data.len)
        ind = np.where(np.array(lst) <= np.floor(val1))[0]
    else:
        val1 = _percentile(data=lst, q=per_dic[-std_value], length=x_data.len)
        ind = np.where(np.array(lst) >= np.floor(val1))[0]

    if return_ind:
        return ind
    else:
        return arr[ind]


def outlier_cooks_distance(x_data: CleanData, y_data: CleanData, plus: bool = True, std_value: int = 2, return_ind: bool = False):
    x = sm.add_constant(data=x_data.data)
    y = y_data.data
    model = sm.OLS(y, x).fit()
    np.set_printoptions(suppress=True)
    influence = model.get_influence()
    cooks = influence.cooks_distance

    if plus:
        val1 = _percentile(data=cooks[0], q=per_dic[std_value], length=x_data.len)
        ind = np.where(cooks[0] <= val1)[0]
    else:
        val1 = _percentile(data=cooks[0], q=per_dic[-std_value], length=x_data.len)
        ind = np.where(cooks[0] >= val1)[0]

    if return_ind:
        return ind
    else:
        return np.array(x_data.data)[ind]
