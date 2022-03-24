"""
Scatter plot class.

Usage:
 ./plot/scatter.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def insert_every(L, char, every):
    """generates items composed of L-items interweaved with char every-so-many items"""
    for i in range(len(L)):
        yield L[i]
        if (i + 1) % every == 0:
            yield char


# fonts = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
# location = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',  'center right', 'lower center', 'upper center', 'center']


@dataclass
class Scatter:
    """

    Class for plotting scatter plots.

    :param data: Input data.
    :type data: pd.DataFrame,
    :param limit: Limit the length of data. *Optional*
    :type limit: int
    :param label_lst: List of labels to include, if None will include all columns. *Optional*
    :type label_lst: List[str]
    :param color_lst: List of colors to graph, needs to be same length as label_lst. *Optional*
    :type color_lst: List[str]
    :param normalize_x: List of columns to normalize. *Optional*
    :type normalize_x: List[str]
    :param regression_line:  If included, requires a column str or List[str], default = None. *Optional*
    :type regression_line: List[str]
    :param regression_line_color: Color of regression line, default = 'red'. *Optional*
    :type regression_line_color: str
    :param regression_line_lineweight: Regression lineweight, default = 2.0. *Optional*
    :type regression_line_lineweight: float
    :param running_mean_x: List of columns to calculate running mean. *Optional*
    :type running_mean_x: List[str]
    :param running_mean_value: List of columns to calculate running mean. *Optional*
    :type running_mean_value: Optional[int] = 50,
    :param cumulative_mean_x: List of columns to calculate cumulative mean. *Optional*
    :type cumulative_mean_x: List[str]
    :param fig_size: default = (10, 7), *Optional*
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
    :param title: Graph title, default = 'Scatter Plot'. *Optional*
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
    :param compare_two: If given will return a scatter comparing two variables, default is None. *Optional*
    :type compare_two: List[str]
    :param y_limit: If given will limit the y axis.
    :type y_limit: List[float]
    :example: *None*
    :note: *None*

    """

    __slots__ = ("ax")

    def __init__(self,
                 data: pd.DataFrame,
                 limit: Optional[int] = None,
                 label_lst: Optional[List[str]] = None,
                 color_lst: Optional[List[str]] = None,
                 normalize_x: Optional[List[str]] = None,
                 regression_line: Optional[List[str]] = None,
                 regression_line_color: Optional[str] = 'r',
                 regression_line_lineweight: Optional[float] = 2.0,
                 running_mean_x: Optional[List[str]] = None,
                 running_mean_value: Optional[int] = 50,
                 cumulative_mean_x: Optional[List[str]] = None,
                 fig_size: Optional[tuple] = (10, 7),
                 ylabel: Optional[str] = None,
                 ylabel_color: Optional[str] = 'black',
                 ylabel_size: Optional[str] = 'medium',
                 xlabel: Optional[str] = None,
                 xlabel_color: Optional[str] = 'black',
                 xlabel_size: Optional[str] = 'medium',
                 title: Optional[str] = 'Scatter Plot',
                 title_size: Optional[str] = 'xx-large',
                 grid: Optional[bool] = True,
                 grid_alpha: Optional[float] = 0.75,
                 grid_dash_sequence: Optional[tuple] = (3, 3),
                 grid_lineweight: Optional[float] = 0.5,
                 legend_fontsize: Optional[str] = 'medium',
                 legend_transparency: Optional[float] = 0.75,
                 legend_location: Optional[str] = 'lower right',
                 compare_two: Optional[List[str]] = None,
                 y_limit: Optional[List[float]] = None
                 ):

        if label_lst is None:
            label_lst = list(data.columns)

        if color_lst is None:
            n = len(label_lst)
            if n == 1:
                color_lst = ['tab:orange']
                regression_line_color = 'tab:blue'
            else:
                color_lst = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]

        fig, ax = plt.subplots(figsize=fig_size)

        if limit:
            data = data[:limit]

        x_axis = range(len(data))

        if compare_two:
            label_lst = [compare_two[1]]
            x_axis = data[compare_two[0]]

        count = 0
        for ind in label_lst:
            d = data[ind]

            ax.scatter(x=x_axis, y=d, color=color_lst[count], label=ind)

            if ind in regression_line:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_axis, d)

                if len(label_lst) == 1:
                    c = regression_line_color
                else:
                    c = color_lst[count]

                ax.plot(x_axis, intercept + slope * x_axis, color=c, label=ind+'_ols_'+str(round(slope, 2)),
                        linestyle='--', linewidth=regression_line_lineweight)
            count += 1

        ax.set_ylabel(ylabel, color=ylabel_color, fontsize=ylabel_size)
        ax.tick_params(axis='y', labelcolor=ylabel_color)
        ax.set_title(title, fontsize=title_size)

        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)

        ax.set_xlabel(xlabel, color=xlabel_color, fontsize=xlabel_size)
        ax.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)

        if y_limit:
            ax.set_ylim(bottom=y_limit[0], top=y_limit[1])

        self.ax = ax

    def __repr__(self):
        return 'Scatter Plot'
