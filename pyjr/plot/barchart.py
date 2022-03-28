"""
Barchart plot class.

Usage:
 ./plot/table.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils.tools import _to_metatype
from pyjr.utils.base import _mean, _std, _sum, _median


def insert_every(L, char, every):
    """generates items composed of L-items interweaved with char every-so-many items"""
    for i in range(len(L)):
        yield L[i]
        if (i + 1) % every == 0:
            yield char


# fonts = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
# location = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',  'center right', 'lower center', 'upper center', 'center']


@dataclass
class Bar:

    __slots__ = "ax"

    def __init__(self,
                 data: Union[pd.DataFrame, Data, PreProcess, List[Union[Data, PreProcess]]],
                 value_string: str = 'sum',
                 label_lst: Optional[Union[List[str], Tuple[str]]] = None,
                 vert_hor: bool = True,
                 xlabel: Optional[str] = 'Names',
                 xlabel_size: Optional[str] = 'medium',
                 ylabel: Optional[str] = 'Values',
                 ylabel_size: Optional[str] = 'medium',
                 title: Optional[str] = 'Bar Chart',
                 title_size: Optional[str] = 'xx-large',
                 limit: Optional[Union[List[int], Tuple[int]]] = None,
                 include_mu: bool = False,
                 mu_color: Optional[str] = 'r',
                 color_lst: list = None,
                 xtick_rotation: int = -90,
                 xtick_size: Optional[str] = 'small',
                 grid: bool = True,
                 grid_alpha: float = 0.75,
                 grid_lineweight: float = 0.5,
                 grid_dash_sequence: tuple = (1, 3),
                 fig_size: Optional[tuple] = (10, 7)):
        # Parse input data
        if isinstance(data, (Data, PreProcess)):
            if label_lst is None:
                label_lst = _to_metatype(data=data.name, dtype='list')
            data = data.dataframe()
        elif isinstance(data, pd.DataFrame):
            if label_lst is None:
                label_lst = _to_metatype(data=data.columns, dtype='list')
        elif isinstance(data, list):
            dic = {}
            for d in data:
                if isinstance(d.name, (list, tuple)):
                    for ind, val in enumerate(d.name):
                        dic[val.name] = val.data[:, ind]
                else:
                    dic[d.name] = d.data
            data = pd.DataFrame.from_dict(dic)
            label_lst = _to_metatype(data=data.columns, dtype='list')

        if limit:
            data = data[limit[0]:limit[1]]

        # Get values
        value_lst = []
        for key in label_lst:
            val = _to_metatype(data=data[key], dtype='list')
            if value_string == 'sum':
                val = _sum(data=val)
            elif value_string == 'mean':
                val = _mean(data=val)
            elif value_string == 'median':
                val = _median(data=val)
            elif value_string == 'std':
                val = _std(data=val)
            value_lst.append(val)

        # Get colors
        if color_lst is None:
            color_lst = ['tab:orange' for i in range(label_lst.__len__())]
        elif color_lst == 'gradient':
            color_lst = [plt.get_cmap('viridis')(1. * i / label_lst.__len__()) for i in range(label_lst.__len__())]
        elif isinstance(color_lst, str) and color_lst != 'gradient':
            color_lst = [color_lst for i in range(label_lst.__len__())]

        if include_mu:
            value_lst.append(_mean(value_lst))
            label_lst.append('mu')
            color_lst.append(mu_color)

        # Start plot
        fig, ax = plt.subplots(figsize=fig_size)

        # Plot values
        if vert_hor:
            ax.bar(label_lst, value_lst, color=color_lst)
        else:
            ax.barh(label_lst, value_lst, color=color_lst)
            if ylabel == 'Values':
                xlabel = 'Values'
                ylabel = 'Names'

        plt.ylabel(ylabel, fontsize=ylabel_size)
        plt.xlabel(xlabel, fontsize=xlabel_size)
        plt.title(title, fontsize=title_size)

        # Add grid
        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)

        self.ax = ax

    def __repr__(self):
        return 'Bar Chart Plot'
