"""
Stand-alone functions.

Usage:
 ./utils/simple.py

Author:
 Peter Rigali - 2022-03-19
"""
import statsmodels.api as sm
from typing import Union, Optional
from pandas import Series
import numpy as np
from pyjr.classes.data import Data
from pyjr.utils._tools.clean import _prep, _type, _mtype, _rval, _rnan, _replace_na, _empty
from pyjr.utils._tools.math import _perc, _var, _mean, _sum
from pyjr.utils._tools.array import _stack
from pyjr.utils._tools.general import _dis, _cent, _unique_values


def _clean(data, dtype, na):
    if na is None:
        na = _rval(d=data, na='zero')
    else:
        na = _rval(d=data, na=na)
    data = _replace_na(d=data, rval=na)
    if dtype is None:
        dtype = np.min_scalar_type(data).type
    else:
        type_tup = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
                    np.float32, np.float64, np.float_, np.str_, np.int_)
        test_dic = {i.__name__: True for i in type_tup}
        if dtype in test_dic:
            dtype = {i.__name__: i for i in type_tup}[dtype]
            if dtype in {np.float_: True, np.int_: True, np.bool_: True}:
                data = dtype(data).tolist()
            else:
                data = np.array(data).astype(dtype).tolist()
        else:
            raise AttributeError("Must use numpy dtypes")
    return data


def _one_hot_encode(func):
    def wrapper(*args, **kwargs):
        data, dtype, na = func(*args, **kwargs)
        data = _clean(data, dtype, na)
        unique = _unique_values(data=data, count=False)
        arr = np.zeros((len(data), len(unique)))
        for ind in range(len(unique)):
            arr[:, ind] = [1.0 if str(ind) == str(val) else 0.0 for val in data]
        return arr
    return wrapper


@_one_hot_encode
def oneHotEncode(data: list, dtype: str = "str_", na: str = None):
    """One hot encode a list of data"""
    return (data, dtype, na)


def calc_gini(d: Union[list, np.ndarray, Series],
              na: str = 'none',
              dtype: str = 'float') -> Union[float, int]:
    """

    Calculate the Gini Coef for a list.

    :param data: Input data.
    :type data: list, np.ndarray, or pd.Series
    :return: Gini value.
    :rtype: float
    :example:
        >>> lst = [4.3, 5.6]
        >>> calc_gini(data=lst, val=4, remainder=True) # 0.05445544554455435
    :note: The larger the gini coef, the more consolidated the chips on the table are to one person.
    """
    new_data = _prep(d=d, dtype=dtype, na=na)
    sorted_list = sorted(new_data)
    height, area = 0.0, 0.0
    for value in sorted_list:
        height += value
        area += height - value / 2.0
    fair_area = height * len(new_data) / 2.0
    return _type(v=(fair_area - area) / fair_area, dtype=dtype)


def outlier_std(data, plus: bool = True, std_value: int = 2, return_ind: bool = False) -> np.ndarray:
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
    per_dic = {-3: 0.001, -2: 0.023, -1: 0.159, 0: 0.50, 1: 0.841, 2: 0.977, 3: 0.999}
    new_data = np.array(data.data)
    if data.min >= 0:
        if plus:
            ind = np.where(new_data <= _perc(d=data.data, q=per_dic[std_value]))[0]
        else:
            ind = np.where(new_data >= _perc(d=data.data, q=per_dic[-std_value]))[0]
    else:
        if plus:
            ind = np.where(new_data <= data.mean + data.std * std_value)[0]
        else:
            ind = np.where(new_data >= data.mean - data.std * std_value)[0]

    if return_ind:
        return ind
    else:
        return new_data[ind]


def outlier_var(data: Data, plus: Optional[bool] = True, std_value: int = 2,
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
    per_dic = {-3: 0.001, -2: 0.023, -1: 0.159, 0: 0.50, 1: 0.841, 2: 0.977, 3: 0.999}
    lst = data.data
    temp_var = _var(d=lst, dof=1)
    dev_based = np.array([temp_var - _var(d=np.delete(lst, i), dof=1) for i, j in enumerate(lst)])

    if plus:
        q = _perc(d=lst, q=per_dic[std_value])
        ind = np.where(dev_based <= q)[0]
    else:
        q = _perc(d=lst, q=per_dic[-std_value])
        ind = np.where(dev_based >= q)[0]

    if return_ind:
        return ind
    else:
        return np.array(lst)[ind]


def outlier_regression(x_data: Data, y_data: Data, plus: Optional[bool] = True, std_value: Optional[int] = 2,
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
    per_dic = {-3: 0.001, -2: 0.023, -1: 0.159, 0: 0.50, 1: 0.841, 2: 0.977, 3: 0.999}
    arr = _stack(np.array(x_data.data), np.array(y_data.data), False)
    ran = np.array(range(x_data.len))
    mu_y = np.zeros(len(arr) - 1)
    line_ys = []
    for i, j in enumerate(arr):
        xx, yy = np.delete(arr[:, 0], i), np.delete(arr[:, 1], i)
        w1 = (np.cov(xx, yy, ddof=1) / _var(xx, dof=1))[0, 1]
        new_y = w1 * ran[:-1] + (-1 * _mean(xx) * w1 + _mean(yy))
        mu_y = (mu_y + new_y) / 2
        line_ys.append(new_y)

    reg_based = np.array([np.mean(np.square(mu_y - j)) for i, j in enumerate(line_ys)])
    if plus:
        threshold = _perc(d=reg_based, q=per_dic[std_value])
        ind = np.where(reg_based <= threshold)[0]
    else:
        threshold = _perc(d=reg_based, q=per_dic[-std_value])
        ind = np.where(reg_based >= threshold)[0]

    if return_ind:
        return ind
    else:
        return arr[ind]


def outlier_distance(x_data: Data, y_data: Data, plus: Optional[bool] = True, std_value: int = 2,
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
    per_dic = {-3: 0.001, -2: 0.023, -1: 0.159, 0: 0.50, 1: 0.841, 2: 0.977, 3: 0.999}
    arr = _stack(np.array(x_data.data), np.array(y_data.data), False)
    cent_other = _cent(arr[:, 0], arr[:, 1])
    ran = range(0, x_data.len)
    x_y_other_centers = np.array([_dis(_cent(x_lst=[arr[i][0]], y_lst=[arr[i][1]]), cent_other) for i in ran])

    if plus:
        x_y_other_centers_std = _perc(d=x_y_other_centers, q=per_dic[std_value])
        ind = np.where(x_y_other_centers <= x_y_other_centers_std)[0]
    else:
        x_y_other_centers_std = _perc(d=x_y_other_centers, q=per_dic[-std_value])
        ind = np.where(x_y_other_centers >= x_y_other_centers_std)[0]

    if return_ind:
        return ind
    else:
        return arr[ind]


def outlier_hist(data: Data, plus: Optional[bool] = True, std_value: int = 2, return_ind: bool = False) -> np.ndarray:
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
    per_dic = {-3: 0.001, -2: 0.023, -1: 0.159, 0: 0.50, 1: 0.841, 2: 0.977, 3: 0.999}
    arr = np.array(data.data)
    n, b = np.histogram(arr, bins='sturges')

    if plus:
        qn = _perc(d=data.data, q=per_dic[std_value])
        ind = np.where(n <= qn)[0]
        bin_edges = np.array([(b[i], b[i + 1]) for i in range(len(b) - 1)])[ind]
    else:
        qn = _perc(d=data.data, q=per_dic[-std_value])
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


def outlier_knn(x_data: Data, y_data: Data, plus: Optional[bool] = True, std_value: int = 2,
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
    per_dic = {-3: 0.001, -2: 0.023, -1: 0.159, 0: 0.50, 1: 0.841, 2: 0.977, 3: 0.999}
    arr = _stack(np.array(x_data.data), np.array(y_data.data), False)
    ran = range(0, x_data.len)
    test_centers = (_cent([arr[ind, 0]], [arr[ind, 1]]) for ind in ran)
    distances = [_dis(c1=i, c2=j) for i in test_centers for j in test_centers]

    if plus:
        threshold = _perc(d=distances, q=per_dic[std_value])
        count_dic = {}
        for i, j in enumerate(arr):
            temp = arr[i, :] <= threshold
            v = [1 for i in temp if i == True]
            if _empty(d=v) is False:
                count_dic[i] = _sum(v)
    else:
        threshold = _perc(d=distances, q=per_dic[-std_value])
        count_dic = {}
        for i, j in enumerate(arr):
            temp = arr[i, :] >= threshold
            count_dic[i] = _sum([1 for i in temp if i == True])

    lst = []
    for val in _mtype(d=count_dic.values()):
        if isinstance(val, list):
            for val1 in val:
                lst.append(val1)
        else:
            lst.append(val)

    if plus:
        val1 = _perc(d=lst, q=per_dic[std_value])
        ind = np.where(np.array(lst) <= np.floor(val1))[0]
    else:
        val1 = _perc(d=lst, q=per_dic[-std_value])
        ind = np.where(np.array(lst) >= np.floor(val1))[0]

    if return_ind:
        return ind
    else:
        return arr[ind]


def outlier_cooks_distance(x_data: Data, y_data: Data, plus: bool = True, std_value: int = 2, return_ind: bool = False):
    per_dic = {-3: 0.001, -2: 0.023, -1: 0.159, 0: 0.50, 1: 0.841, 2: 0.977, 3: 0.999}
    x = sm.add_constant(data=x_data.data)
    y = y_data.data
    model = sm.OLS(y, x).fit()
    np.set_printoptions(suppress=True)
    influence = model.get_influence()
    cooks = influence.cooks_distance

    if plus:
        val1 = _perc(d=cooks[0], q=per_dic[std_value])
        ind = np.where(cooks[0] <= val1)[0]
    else:
        val1 = _perc(d=cooks[0], q=per_dic[-std_value])
        ind = np.where(cooks[0] >= val1)[0]

    if return_ind:
        return ind
    else:
        return np.array(x_data.data)[ind]
