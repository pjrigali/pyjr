"""
Histogram plot class.

Usage:
 ./plot/histogram.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils._tools.clean import _mtype
from pyjr.utils._tools.math import _mean, _std


@dataclass
class Histogram:
    """

    Class for plotting histograms.

    :param data: Input data.
    :type data: Either Data(pyjr class), PreProcess(pyjr class), list of the latter or a dict, pd.DataFrame.
    :param limit: Selection of rows to include.
    :type limit: list or tuple of int's.
    :param label_lst: Columns or names to focus on.
    :type label_lst: list or tuple of str's.
    :param color_lst: List or tuple of colors to use in plot.
    :type color_lst: List or tuple
    :param include_norm: Include norm. If included, requires a column str, default = None. *Optional*
    :type include_norm: str
    :param norm_color: Norm color, default = 'red'. *Optional*
    :type norm_color: str
    :param norm_lineweight: Norm lineweight, default = 1.0. *Optional*
    :type norm_lineweight: float
    :param norm_ylabel: Norm Y axis label. *Optional*
    :type norm_ylabel: str
    :param norm_legend_location: Location of norm legend, default = 'upper right'. *Optional*
    :type norm_legend_location: str
    :param fig_size: default = (10, 7), *Optional*
    :type fig_size: tuple
    :param bins: Way of calculating bins, default = 'sturges'. *Optional*
    :type bins: str
    :param hist_type: Type of histogram, default = 'bar'. *Optional*
    :type hist_type: str
    :param stacked: If True, will stack histograms, default = False. *Optional*
    :type stacked: bool
    :param ylabel: Y axis label. *Optional*
    :type ylabel: str
    :param ylabel_color: Y axis label color, default = 'black'. *Optional*
    :type ylabel_color: str
    :param ylabel_size: Y label size, default = 'medium'. *Optional*
    :type ylabel_size: str
    :param ytick_rotation:
    :type ytick_rotation: int = 0,
    :param xlabel: X axis label. *Optional*
    :type xlabel: str
    :param xlabel_color: X axis label color, default = 'black'. *Optional*
    :type xlabel_color: str
    :param xlabel_size: X label size, default = 'medium'. *Optional*
    :type xlabel_size: str
    :param xtick_rotation:
    :type xtick_rotation: int = 0,
    :param title: Title of plt.
    :type title: str
    :param title_size: Size of title font. *Default is xx-large*
    :type title_size: str
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
    :param legend_fontsize: Legend fontsize, default = 'medium'. *Optional*
    :type legend_fontsize: str
    :param legend_transparency: Legend transparency, default = 0.75. *Optional*
    :type legend_transparency: float
    :param legend_location: legend location, default = 'lower right'. *Optional*
    :type legend_location: str
    :param show: Whether to print the plot. *Default is False.*
    :type show: bool
    :example: *None*
    :note:
        fonts can be: {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        location can be: {'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',
                          'center right', 'lower center', 'upper center', 'center'}
    """
    __slots__ = ("ax", "ax1")

    def __init__(self,
                 data: Union[pd.DataFrame, Data, PreProcess, List[Union[Data, PreProcess]]],
                 color_lst: Union[List[str], Tuple[str]] = None,
                 label_lst: Union[List[str], Tuple[str]] = None,
                 limit: Union[List[int], Tuple[int]] = None,
                 include_norm: str = None,
                 norm_color: str = 'r',
                 norm_lineweight: float = 1.0,
                 norm_ylabel: str = None,
                 norm_legend_location: str = 'upper right',
                 fig_size: tuple = (10, 7),
                 bins: str = 'sturges',
                 hist_type: str = 'bar',
                 stacked: bool = False,
                 ylabel: str = None,
                 ylabel_color: str = 'black',
                 ylabel_size: str = 'medium',
                 ytick_rotation: int = 0,
                 xlabel: str = None,
                 xlabel_color: str = 'black',
                 xlabel_size: str = 'medium',
                 xtick_rotation: int = 0,
                 title: str = 'Histogram',
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

        # Start plot
        fig, ax = plt.subplots(figsize=fig_size)

        if limit:
            data = data[limit[0]:limit[1]]

        # Get plots
        count = 0
        for ind in label_lst:
            ax.hist(data[ind], bins=bins, color=color_lst[count], label=ind, stacked=stacked, histtype=hist_type)
            count += 1
        ax.set_ylabel(ylabel, color=ylabel_color, fontsize=ylabel_size)
        ax.tick_params(axis='y', labelcolor=ylabel_color, rotation=ytick_rotation)
        ax.set_title(title, fontsize=title_size)

        # Add grid
        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)

        ax.set_xlabel(xlabel, color=xlabel_color, fontsize=xlabel_size)
        ax.tick_params(axis='x', labelcolor=ylabel_color, rotation=xtick_rotation)
        ax.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)

        # Plot normal curve
        ax1 = None
        if include_norm:
            d = _mtype(d=data[include_norm], dtype='list')
            _mu, _s = norm.fit(np.random.normal(_mean(d=d), _std(d=d), d.__len__()))
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            ax1 = ax.twinx()
            ax1.plot(x, norm.pdf(x, _mu, _s), color=norm_color, linewidth=norm_lineweight, linestyle='--',
                     label="{} Fit Values: mu {:.2f} and std {:.2f}".format(include_norm, _mu, _s))
            ax1.set_ylabel(norm_ylabel, color=norm_color)
            ax1.tick_params(axis='y', labelcolor=norm_color)
            ax1.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=norm_legend_location, frameon=True)

        self.ax = ax
        self.ax1 = ax1

        if show:
            plt.show()

    def __repr__(self):
        return 'Histogram Plot'
