"""
DataFrame functions.

Usage:
 ./utils/_tools/dataframe.py

Author:
 Peter Rigali - 2022-03-30
"""
from typing import Union
from pandas import DataFrame
from pyjr.utils._tools.clean import _mtype


def slc(d: DataFrame, c: str, v: Union[float, int, str, object]) -> DataFrame:
    """
    Select function filters a dataframe using col and a value.

    :param d: Selected DataFrame
    :type d: pd.DataFrame
    :param c: Column
    :type c: str
    :param v: Value to look for.
    :type v: float, int, str, or object
    :return: Returns part of DataFrame matching value within desired column.
    :rtype: pd.DataFrame


    """
    return d[d[c] == v]


def ilc(d: DataFrame, i) -> DataFrame:
    """
    iloc function filters a dataframe using a list of values.

    :param d: Selected DataFrame
    :type d: pd.DataFrame
    :param i: list of indexes to filter rows by.
    :type i: List like objects
    :return: Returns part of DataFrame matching the input index list.
    :rtype: pd.DataFrame


    """
    return d.iloc[_mtype(d=i, dtype='list')]
