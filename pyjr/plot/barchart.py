"""
Barchart plot class.

Usage:
 ./plot/table.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils.tools.clean import _mtype
from pyjr.utils.tools.math import _mean, _std, _sum, _median


@dataclass
class Bar:
    """

    Builds Bar Chart Class. Used for plotting data.

    :param data: Input data.
    :type data: Either Data(pyjr class), PreProcess(pyjr class), list of the latter or a dict, pd.DataFrame.
    :param value_string: Input data conversion, {sum, mean, median, std}. *Default is sum.*
    :type value_string: str.
    :param label_lst: Columns or names to focus on.
    :type label_lst: list or tuple of str's.
    :param vert_hor: True is vertical, False is horizontal. *Default is True.*
    :type vert_hor: bool.
    :param xlabel: Plot X axis label.
    :type xlabel: str
    :param xlabel_size: Plot X axis font size. *Default is medium*
    :type xlabel_size: str
    :param ylabel: Plot Y axis label.
    :type ylabel: str
    :param ylabel_size: Plot Y axis font size. *Default is medium*
    :type ylabel_size: str
    :param title: Title of plt.
    :type title: str
    :param title_size: Size of title font. *Default is xx-large*
    :type title_size: str
    :param limit: Selection of rows to include.
    :type limit: list or tuple of int's.
    :param include_mu: Include the mean of the values. *Default is True*
    :type include_mu: bool
    :param mu_color: Color of mu data. *Default is red.*
    :type mu_color: str
    :param color_lst: List or tuple of colors to use in plot.
    :type color_lst: List or tuple
    :param grid: Whether to include a grid. *Default is True.*
    :type grid: bool
    :param grid_alpha: Transparency of grid lines. *Default is 0.75'*
    :type grid_alpha: float
    :param grid_lineweight: Width of gridlines. *Default is 0.5.*
    :type grid_lineweight: float
    :param grid_dash_sequence: Tuple of spaces and lines in grid. *Default is (1, 3)*
    :type grid_dash_sequence: tuple
    :param fig_size: Width and height of figure. *Default is (10, 7).*
    :type fig_size: tuple
    :param show: Whether to print the plot. *Default is False.*
    :type show: bool
    :example: *None*
    :note:
        fonts can be: {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        location can be: {'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',
                          'center right', 'lower center', 'upper center', 'center'}

    """
    __slots__ = "ax"

    def __init__(self,
                 data: Union[pd.DataFrame, Data, PreProcess, List[Union[Data, PreProcess]], dict],
                 value_string: str = 'sum',
                 label_lst: Union[List[str], Tuple[str]] = None,
                 vert_hor: bool = True,
                 xlabel: str = 'Names',
                 xlabel_size: str = 'medium',
                 ylabel: str = 'Values',
                 ylabel_size: str = 'medium',
                 title: str = 'Bar Chart',
                 title_size: str = 'xx-large',
                 limit: Union[List[int], Tuple[int]] = None,
                 include_mu: bool = False,
                 mu_color: str = 'r',
                 color_lst: Union[list, tuple] = None,
                 grid: bool = True,
                 grid_alpha: float = 0.75,
                 grid_lineweight: float = 0.5,
                 grid_dash_sequence: tuple = (1, 3),
                 fig_size: tuple = (10, 7),
                 show: bool = False,
                 ):
        # Parse input data
        dic = False
        if isinstance(data, (Data, PreProcess)):
            if label_lst is None:
                label_lst = _mtype(d=data.name, dtype='list')
            data = data.dataframe()
        elif isinstance(data, pd.DataFrame):
            if label_lst is None:
                label_lst = _mtype(d=data.columns, dtype='list')
        elif isinstance(data, list):
            _dic = {}
            for d in data:
                if isinstance(d.name, (list, tuple)):
                    for ind, val in enumerate(d.name):
                        _dic[val.name] = val.data[:, ind]
                else:
                    _dic[d.name] = d.data
            data = pd.DataFrame.from_dict(_dic)
            label_lst = _mtype(d=data.columns, dtype='list')
        elif isinstance(data, dict):
            value_lst = _mtype(d=data.values(), dtype='list')
            label_lst = _mtype(d=data.keys(), dtype='list')
            dic = True

        if dic is False:
            if limit:
                data = data[limit[0]:limit[1]]
            # Get values
            value_lst = []
            for key in label_lst:
                val = _mtype(d=data[key], dtype='list')
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

        if show:
            plt.show()

    def __repr__(self):
        return 'Bar Chart Plot'
