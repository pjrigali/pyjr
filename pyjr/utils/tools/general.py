"""
General functions.

Usage:
 ./utils/tools/general.py

Author:
 Peter Rigali - 2022-03-30
"""
from typing import Union, List
from pyjr.utils.tools.clean import _prep
import numpy as np


def _unique_values(data: Union[list, tuple], count: bool = False):
    """
    Finds unique values from a list.

    :param data: Input data.
    :type data: list or tuple.
    :return: Returns either a list or dict.
    :note: *None*
    """
    if count:
        unique = set(data)
        return {i: data.count(i) for i in unique}
    else:
        return tuple(set(data))


def _add_constant(data: Union[list, tuple, np.ndarray]) -> np.ndarray:
    """Add a column of ones to a list, tuple or np.ndarray"""
    if isinstance(data, (tuple, list)):
        arr = np.ones((data.__len__(), 2))
    elif isinstance(data, np.ndarray):
        arr = np.ones((data.shape[0], 2))
    arr[:, 1] = data
    return arr


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


def _dis(c1: List[float], c2: List[float]) -> float:
    """

    Calculate Distance between two centroids.

    :param c1: An x, y coordinate representing a centroid.
    :type c1: List[float]
    :param c2: An x, y coordinate representing a centroid.
    :type c2: List[float]
    :returns: A distance measurement.
    :rtype: float
    :example: *None*
    :note: *None*

    """
    return round(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2), 4)


def tru_dic(d, na: str = 'none', dtype: str = 'none') -> dict:
    """

    Returns a dictionary from a list with True as the values.

    :param d: List, array, series like object.
    :param na: How to handle na values in the list object.
    :type na: str
    :param dtype: What type to convert the list values to.
    :type dtype: str
    :example: *None*
    :note: *None*

    """
    d = _prep(d=d, mtype='tuple', na=na, dtype=dtype)
    return {i: True for i in d}
