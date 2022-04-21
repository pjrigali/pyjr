"""
General functions.

Usage:
 ./utils/tools/datetime.py

Author:
 Peter Rigali - 2022-04-20
"""
from typing import Union
from datetime import datetime, timedelta
import holidays
from pyjr.utils.tools.clean import _mtype
HOLIDAYS_US = holidays.US()


def to_datetimes(d, fmt: str = '%m/%d/%y', mtype: str = 'tuple', from_timestamps: bool = False) -> Union[list, tuple]:
    if from_timestamps is True:
        return _mtype(d=[datetime.fromtimestamp(i) for i in d], dtype=mtype)
    else:
        return _mtype(d=[datetime.strptime(i, fmt) for i in d], dtype=mtype)


def to_date_strs(d, mtype: str = 'tuple') -> Union[list, tuple]:
    lst = []
    for date in d:
        s = str(date.year)
        if len(str(date.month)) == 2:
            s += '_' + str(date.month)
        else:
            s += '_0' + str(date.month)
        if len(str(date.day)) == 2:
            s += '_' + str(date.day)
        else:
            s += '_0' + str(date.day)
        lst.append(s)
    return _mtype(d=lst, dtype=mtype)


def group_by(d: Union[list, tuple], method: str = 'y', mtype: str = 'tuple') -> dict:
    if method == 'y':
        dic, c_dic = {}, {}
        for date in d:
            if date in c_dic:
                dic[date.year].append(date)
            else:
                c_dic[date.year], dic[date.year] = True, [date]
    elif method == 'm':
        dic, c_dic = {}, {}
        for date in d:
            if date in c_dic:
                dic[date.month].append(date)
            else:
                c_dic[date.month], dic[date.month] = True, [date]
    elif method == 'y_m':
        dic, c_dic = {}, {}
        for date in d:
            if date in c_dic:
                dic[str(date.year) + '_' + str(date.month)].append(date)
            else:
                c_dic[str(date.year) + '_' + str(date.month)], dic[str(date.year) + '_' + str(date.month)] = True, [date]
    else:
        raise AttributeError('Must be {y, m, y_m}')

    return {key: _mtype(d=val, dtype=mtype) for key, val in dic.items()}


def unique_dates(d: Union[list, tuple], method: str = 'y', mtype: str = 'tuple') -> Union[list, tuple]:
    if method == 'y':
        lst, c_dic = [], {}
        for date in d:
            if date.year not in c_dic:
                lst.append(date.year)
                c_dic[date.year] = True
    elif method == 'm':
        lst, c_dic = [], {}
        for date in d:
            if date.month not in c_dic:
                lst.append(date.month)
                c_dic[date.month] = True
    elif method == 'y_m':
        lst, c_dic = [], {}
        for date in d:
            if str(date.year) + '_' + str(date.month) not in c_dic:
                lst.append(str(date.year) + '_' + str(date.month))
                c_dic[str(date.year) + '_' + str(date.month)] = True
    else:
        raise AttributeError('Must be {y, m, y_m}')

    return _mtype(d=lst, dtype=mtype)


def next_business_day(date, days: int = 1):
    d = date + timedelta(days=days)
    while d.weekday() in holidays.WEEKEND or d in HOLIDAYS_US:
        d += timedelta(days=days)
    return d


def to_dt(y: int, m: int = 1, d: int = 1):
    return datetime(year=y, month=m, day=d)
