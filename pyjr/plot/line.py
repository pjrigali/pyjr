"""
Line plot class.

Usage:
 ./plot/line.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils.tools import _to_metatype


def insert_every(L, char, every):
    """generates items composed of L-items interweaved with char every-so-many items"""
    for i in range(len(L)):
        yield L[i]
        if (i + 1) % every == 0:
            yield char


# fonts = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
# location = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',  'center right', 'lower center', 'upper center', 'center']


@dataclass
class Line:
    """

    Class for plotting line plots.

    :param data: Input data.
    :type data: pd.DataFrame,
    :param limit: Limit the length of data. *Optional*
    :type limit: int
    :param label_lst: List of labels to include, if None will include all columns. *Optional*
    :type label_lst: List[str]
    :param color_lst: List of colors to graph, needs to be same length as label_lst. *Optional*
    :type color_lst: List[str]
    :param fig_size: Figure size, default = (10, 7). *Optional*
    :type fig_size: tuple
    :param ylabel: Y axis label. *Optional*
    :type ylabel: str
    :param ylabel_color: Y axis label color, default = 'black'. *Optional*
    :type ylabel_color: str
    :param ylabel_size: Y label size, default = 'medium'. *Optional*
    :type ylabel_size: str
    :param xlabel: X axis label. *Optional*
    :type xlabel: str
    :param xlabel_color: X axis label color, default = 'black'. *Optional*
    :type xlabel_color: str
    :param xlabel_size: X label size, default = 'medium'. *Optional*
    :type xlabel_size: str
    :param title: Graph title, default = 'Line Plot'. *Optional*
    :type title: str
    :param title_size: Title size, default = 'xx-large'. *Optional*
    :type title_size: str
    :param grid: If True will show grid, default = true. *Optional*
    :type grid: bool
    :param grid_alpha: Grid alpha, default = 0.75. *Optional*
    :type grid_alpha: float
    :param grid_dash_sequence: Grid dash sequence, default = (3, 3). *Optional*
    :type grid_dash_sequence: tuple
    :param grid_lineweight: Grid lineweight, default = 0.5. *Optional*
    :type grid_lineweight: float
    :param legend_fontsize: Legend fontsize, default = 'medium'. *Optional*
    :type legend_fontsize: str
    :param legend_transparency: Legend transparency, default = 0.75. *Optional*
    :type legend_transparency: float
    :param legend_location: legend location, default = 'lower right'. *Optional*
    :type legend_location: str
    :example: *None*
    :note: *None*

    """

    __slots__ = "ax"

    def __init__(self,
                 data: Union[pd.DataFrame, Data, PreProcess, List[Union[Data, PreProcess]]],
                 limit: Optional[Union[List[int], Tuple[int]]] = None,
                 label_lst: Optional[Union[List[str], Tuple[str]]] = None,
                 color_lst: Optional[Union[List[str], Tuple[str]]] = None,
                 fig_size: Optional[tuple] = (10, 7),
                 ylabel: Optional[str] = None,
                 ylabel_color: Optional[str] = 'black',
                 ylabel_size: Optional[str] = 'medium',
                 xlabel: Optional[str] = None,
                 xlabel_color: Optional[str] = 'black',
                 xlabel_size: Optional[str] = 'medium',
                 title: Optional[str] = 'Line Plot',
                 title_size: Optional[str] = 'xx-large',
                 grid: Optional[bool] = True,
                 grid_alpha: Optional[float] = 0.75,
                 grid_dash_sequence: Optional[tuple] = (3, 3),
                 grid_lineweight: Optional[float] = 0.5,
                 legend_fontsize: Optional[str] = 'medium',
                 legend_transparency: Optional[float] = 0.75,
                 legend_location: Optional[str] = 'lower right',
                 ):
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

        # Get colors
        if color_lst is None:
            if label_lst.__len__() <= 3:
                color_lst = ['tab:orange', 'tab:blue', 'tab:green'][:label_lst.__len__()]
            else:
                color_lst = [plt.get_cmap('viridis')(1. * i / label_lst.__len__()) for i in range(label_lst.__len__())]

        # Start Plot
        fig, ax = plt.subplots(figsize=fig_size)

        if limit:
            data = data[limit[0]:limit[1]]

        # Get plots
        count = 0
        for name in label_lst:
            d = data[name]
            ax.plot(d, color=color_lst[count], label=name)
            count += 1
        ax.set_ylabel(ylabel, color=ylabel_color, fontsize=ylabel_size)
        ax.tick_params(axis='y', labelcolor=ylabel_color)
        ax.set_title(title, fontsize=title_size)

        # Add grid
        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)
        ax.set_xlabel(xlabel, color=xlabel_color, fontsize=xlabel_size)
        ax.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)
        self.ax = ax

    def __repr__(self):
        return 'Line Plot'
