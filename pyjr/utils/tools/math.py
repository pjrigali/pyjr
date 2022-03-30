"""
Math functions.

Usage:
 ./utils/tools/math.py

Author:
 Peter Rigali - 2022-03-30
"""
from typing import Union
from pyjr.utils.tools.clean import _mtype, _round
from pyjr.utils.tools.general import _unique_values
from pyjr.utils.tools.dic import _search_dic_values


# Internal Math Functions
def _max(d: Union[list, tuple]) -> float:
    """
    Find the max value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Maximum value.
    rtype: float.
    :note: *None*
    """
    if d.__len__() > 1:
        return max(d)
    return d[0]


def _min(d: Union[list, tuple]):
    """
    Find the min value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Minimum value.
    :rtype: float.
    :note: *None*
    """
    if d.__len__() > 1:
        return min(d)
    return d[0]


def _range(d: Union[list, tuple]) -> float:
    """
    Find the max to min range value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Range value.
    :rtype: float.
    :note: *None*
    """
    if d.__len__() > 1:
        return _max(d=d) - _min(d=d)
    return 0.0


def _mean(d: Union[list, tuple]) -> float:
    """
    Find the mean value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Mean value.
    :rtype: float.
    :note: *None*
    """
    return sum(d) / d.__len__()


def _var(d: Union[list, tuple], dof: int = 1) -> float:
    """
    Find the variance value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :param dof: Desired Degrees of Freedom.
    :type dof: int
    :return: Variance value.
    :rtype: float.
    :note: *None*
    """
    mu = _mean(d=d)
    return sum(((x - mu) ** 2 for x in d)) / (d.__len__() - dof)


def _std(d: Union[list, tuple], dof: int = 1) -> float:
    """
    Find the Standard Deviation value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :param dof: Desired Degrees of Freedom.
    :type dof: int
    :return: Standard Deviation value.
    :rtype: float.
    :note: *None*
    """
    return _var(d=d, dof=dof) ** .5


def _sum(d: Union[list, tuple]) -> float:
    """
    Find the sum value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Sum value.
    rtype: float.
    :note: *None*
    """
    if d.__len__() > 1:
        return sum(d)
    return d[0]


def _med(d: Union[list, tuple]) -> float:
    """
    Find the median value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Mean value.
    :rtype: float.
    :note: *None*
    """
    sorted_lst, lst_len = sorted(d), d.__len__()
    index = (lst_len - 1) // 2
    if lst_len % 2:
        return sorted_lst[index]
    else:
        return _mean(d=[sorted_lst[index]] + [sorted_lst[index + 1]])


def _mod(d: Union[list, tuple]) -> float:
    """
    Find the mode value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Mode value.
    :rtype: float.
    :note: *None*
    """
    count_dic = _unique_values(data=d, count=True)
    count_dic_values = _mtype(d=count_dic.values())
    dic_max = _max(count_dic_values)
    lst = []
    for i in count_dic_values:
        val = _search_dic_values(dic=count_dic, item=dic_max)
        lst.append((val, i))
        # del count_dic[val]
        count_dic_values = _mtype(d=count_dic.values())

    first_val, second_val = lst[0][0], lst[0][1]
    equal_lst = [i[0] for i in lst if second_val == i[1]]
    if equal_lst.__len__() == 1:
        return first_val
    elif equal_lst.__len__() % 2 == 0:
        return _mean(d=equal_lst)
    else:
        return _med(d=equal_lst)


def _skew(d: Union[list, tuple]) -> float:
    """
    Find the skew value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Skew value.
    :rtype: float.
    :note: *None*
    """
    mu, stdn, length = _mean(d=d), _std(d=d, dof=1) ** 3, d.__len__()
    if stdn == 0:
        stdn = mu / 2.0
    return (((_sum(d=[i - mu for i in d]) ** 3) / length) / stdn) * ((length * (length - 1)) ** .5) / (length - 2)


def _kurt(d: Union[list, tuple]) -> float:
    """
    Find the kurtosis value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Kurtosis value.
    :rtype: float.
    :note: *None*
    """
    mu, stdn = _mean(d=d), _std(d=d, dof=1) ** 4
    if stdn == 0:
        stdn = mu / 2
    return (((_sum(d=[i - mu for i in d]) ** 4) / d.__len__()) / stdn) - 3


def _perc(d: Union[list, tuple], q: float) -> float:
    """
    Find the percentile value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :param q: Percentile percent.
    :type q: float.
    :return: Percentile value.
    rtype: float.
    :note: *None*
    """
    d = _round(d=[item * 1000.0 for item in d], v=1)
    ind = _round(d=d.__len__() * q, v=1)
    d.sort()
    for item in d:
        if item >= d[ind]:
            return item / 1000.0


def _percs(d: Union[list, tuple], q_lst: Union[list, tuple] = (0.159, 0.841)):
    """
    Calculate various percentiles for a list.

    :param d: Input data.
    :type d: list or tuple.
    :param q_lst: Desired percentile percents.
    :type q_lst: List of floats.
    :return: A group of stats.
    :note: *None*
    """
    return (_perc(d=d, q=q) for q in q_lst)


def _gini(d: Union[list, tuple]) -> float:
    """
    Find the Gini Coef. value of a list.

    :param d: Input data.
    :type d: list or tuple.
    :return: Gini Coef. value.
    :rtype: float.
    :note: *None*
    """
    sorted_list = sorted(d)
    height, area = 0.0, 0.0
    for value in sorted_list:
        height += value
        area += height - value / 2.0
    fair_area = height * d.__len__() / 2.0
    return (fair_area - area) / fair_area
