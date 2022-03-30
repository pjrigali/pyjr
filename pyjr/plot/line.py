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
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils._tools.clean import _mtype


@dataclass
class Line:
    """

    Class for plotting line plots.

    :param data: Input data.
    :type data: Either Data(pyjr class), PreProcess(pyjr class), list of the latter or a dict, pd.DataFrame.
    :param limit: Selection of rows to include.
    :type limit: list or tuple of int's.
    :param label_lst: Columns or names to focus on.
    :type label_lst: list or tuple of str's.
    :param color_lst: List or tuple of colors to use in plot.
    :type color_lst: List or tuple
    :param fig_size: Figure size. *Default = (10, 7).*
    :type fig_size: tuple
    :param ylabel: Y axis label. *Optional*
    :type ylabel: str
    :param ylabel_color: Y axis label color. *Default = 'black'.*
    :type ylabel_color: str
    :param ylabel_size: Y label size. *Default = 'medium'.*
    :type ylabel_size: str
    :param xlabel: X axis label. *Optional*
    :type xlabel: str
    :param xlabel_color: X axis label color.  *Default = 'black'.*
    :type xlabel_color: str
    :param xlabel_size: X label size. *Default = 'medium'.*
    :type xlabel_size: str
    :param title: Title of plt.
    :type title: str
    :param title_size: Size of title font. *Default is xx-large*
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
    :note:
        fonts can be: {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        location can be: {'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',
                          'center right', 'lower center', 'upper center', 'center'}

    """
    __slots__ = "ax"

    def __init__(self,
                 data: Union[pd.DataFrame, Data, PreProcess, List[Union[Data, PreProcess]]],
                 limit: Union[List[int], Tuple[int]] = None,
                 label_lst: Union[List[str], Tuple[str]] = None,
                 color_lst: Union[List[str], Tuple[str]] = None,
                 fig_size: tuple = (10, 7),
                 ylabel: Optional[str] = None,
                 ylabel_color: str = 'black',
                 ylabel_size: str = 'medium',
                 xlabel: Optional[str] = None,
                 xlabel_color: str = 'black',
                 xlabel_size: str = 'medium',
                 title: str = 'Line Plot',
                 title_size: str = 'xx-large',
                 grid: bool = True,
                 grid_alpha: float = 0.75,
                 grid_dash_sequence: tuple = (3, 3),
                 grid_lineweight: float = 0.5,
                 legend_fontsize: str = 'medium',
                 legend_transparency: float = 0.75,
                 legend_location: str = 'lower right',
                 show: bool = False,
                 ):
        # Parse input data
        if isinstance(data, (Data, PreProcess)):
            if label_lst is None:
                label_lst = _mtype(d=data.name, dtype='list')
            data = data.dataframe()
        elif isinstance(data, pd.DataFrame):
            if label_lst is None:
                label_lst = _mtype(d=data.columns, dtype='list')
        elif isinstance(data, list):
            dic = {}
            for d in data:
                if isinstance(d.name, (list, tuple)):
                    for ind, val in enumerate(d.name):
                        dic[val.name] = val.data[:, ind]
                else:
                    dic[d.name] = d.data
            data = pd.DataFrame.from_dict(dic)
            label_lst = _mtype(d=data.columns, dtype='list')

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

        if show:
            plt.show()

    def __repr__(self):
        return 'Line Plot'
