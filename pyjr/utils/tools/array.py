"""
Array functions.

Usage:
 ./utils/tools/array.py

Author:
 Peter Rigali - 2022-03-30
"""
from typing import Optional, Union
import numpy as np


def _stack(x_arr: np.ndarray, y_arr: np.ndarray, multi: Optional[bool] = False) -> np.ndarray:
    """

    Stacks x_arr and y_arr.

    :param x_arr: An array to stack.
    :type x_arr: np.ndarray
    :param y_arr: An array to stack.
    :type y_arr: np.ndarray
    :param multi: If True, will stack based on multiple x_arr columns, default is False. *Optional*
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


def _add_column(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Adds a column-arr to an arr"""
    new_arr = np.ones((arr1.shape[0], arr1.shape[1] + 1))
    for i in range(arr1.shape[1]):
        new_arr[:, i] = arr1[:, i]
    new_arr[:, new_arr.shape[1] - 1] = arr2[:, 0]
    return new_arr


def _ts_arr(data: Union[list, tuple]) -> np.ndarray:
    """Builds ts_data required for pyts"""
    arr = np.ones((data.__len__(), 2))
    arr[:, 0] = data
    arr[:, 1] = list(range(data.__len__()))
    return arr.reshape((2, data.__len__()))
