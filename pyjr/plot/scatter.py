"""
Scatter plot class.

Usage:
 ./plot/scatter.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils.tools import _to_metatype


@dataclass
class Scatter:
    """

    Class for plotting scatter plots.

    :param data: Input data.
    :type data: Either Data(pyjr class), PreProcess(pyjr class), list of the latter or a dict, pd.DataFrame.
    :param limit: Selection of rows to include.
    :type limit: list or tuple of int's.
    :param label_lst: Columns or names to focus on.
    :type label_lst: list or tuple of str's.
    :param color_lst: List or tuple of colors to use in plot.
    :type color_lst: List or tuple
    :param regression_line:  If included, requires a column str or List[str], default = None. *Optional*
    :type regression_line: List[str]
    :param regression_line_color: Color of regression line, default = 'red'. *Optional*
    :type regression_line_color: str
    :param regression_line_lineweight: Regression lineweight, default = 2.0. *Optional*
    :type regression_line_lineweight: float
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
    :param compare_two: If given will return a scatter comparing two variables, default is None. *Optional*
    :type compare_two: List[str]
    :param y_limit: If given will limit the y axis.
    :type y_limit: List[float]
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
                 regression_line: Union[List[str], Tuple[str]] = None,
                 regression_line_color: str = None,
                 regression_line_lineweight: float = 2.0,
                 fig_size: tuple = (10, 7),
                 ylabel: str = None,
                 ylabel_color: str = 'black',
                 ylabel_size: str = 'medium',
                 xlabel: str = None,
                 xlabel_color: str = 'black',
                 xlabel_size: str = 'medium',
                 title: str = 'Scatter Plot',
                 title_size: str = 'xx-large',
                 grid: bool = True,
                 grid_alpha: float = 0.75,
                 grid_dash_sequence: tuple = (3, 3),
                 grid_lineweight: float = 0.5,
                 legend_fontsize: str = 'medium',
                 legend_transparency: float = 0.75,
                 legend_location: str = 'lower right',
                 compare_two: Union[Tuple[str], bool] = None,
                 y_limit: Union[list, tuple] = None,
                 show: bool = False,
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

        if regression_line_color is None:
            regression_line_color = 'r'

        # Start plot
        fig, ax = plt.subplots(figsize=fig_size)

        if limit:
            data = data[limit[0]:limit[1]]

        # Get compare two
        if compare_two is True:
            if _to_metatype(data=data.columns, dtype='tuple').__len__() >= 2:
                if ylabel is None:
                    ylabel = label_lst[1]
                if xlabel is None:
                    xlabel = label_lst[0]
                x_axis = data[label_lst[0]]
                label_lst = [label_lst[1]]
            else:
                raise AttributeError("If compare_two is True, input data requires two columns of data.")
        elif isinstance(compare_two, (list, tuple)):
            if compare_two.__len__() == 2:
                label_lst = [compare_two[1]]
                x_axis = data[compare_two[0]]
                if ylabel is None:
                    ylabel = compare_two[1]
                if xlabel is None:
                    xlabel = compare_two[0]
            else:
                raise AttributeError("If compare_two is a list, input data requires two columns of data.")
        else:
            x_axis = range(len(data))
            if ylabel is None:
                ylabel = 'Values'
            if xlabel is None:
                xlabel = 'Index'

        # Get plots
        count = 0
        for ind in label_lst:
            d = data[ind]
            ax.scatter(x=x_axis, y=d, color=color_lst[count], label=ind)
            if regression_line is not None and ind in regression_line:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_axis, d)
                ax.plot(x_axis, intercept + slope * x_axis, color=regression_line_color,
                        label=ind+'_ols_'+str(round(slope, 2)), linestyle='--', linewidth=regression_line_lineweight)
            count += 1
        ax.set_ylabel(ylabel, color=ylabel_color, fontsize=ylabel_size)
        ax.tick_params(axis='y', labelcolor=ylabel_color)
        ax.set_title(title, fontsize=title_size)

        # Add grid
        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)

        ax.set_xlabel(xlabel, color=xlabel_color, fontsize=xlabel_size)

        # Add legend
        if compare_two is None:
            ax.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)

        if y_limit:
            ax.set_ylim(bottom=y_limit[0], top=y_limit[1])

        self.ax = ax

        if show:
            plt.show()

    def __repr__(self):
        return 'Scatter Plot'
