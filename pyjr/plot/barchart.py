"""
Barchart plot class.

Usage:
 ./plot/table.py

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
class BarChart:

    __slots__ = ("ax")

    def __init__(self,
                 data: pd.DataFrame,
                 label_lst: Optional[List[str]] = None,
                 xlabel: Optional[str] = 'X Axis',
                 xlabel_size: Optional[str] = 'medium',
                 ylabel: Optional[str] = 'Y Axis',
                 ylabel_size: Optional[str] = 'medium',
                 title: Optional[str] = 'Bar Chart',
                 title_size: Optional[str] = 'xx-large',
                 limit: int = None,
                 spacing: int = None,
                 include_mu: bool = None,
                 mu_color: Optional[str] = 'r',
                 color_lst: list = None,
                 xtick_rotation: int = -90,
                 xtick_size: Optional[str] = 'small',
                 grid: bool = True,
                 grid_alpha: float = 0.75,
                 grid_lineweight: float = 0.5,
                 grid_dash_sequence: tuple = (1, 3),
                 legend_transparency: float = 0.5,
                 legend_fontsize: Optional[str] = 'medium',
                 legend_location: Optional[str] = 'upper right',
                 fig_size: Optional[tuple] = (10, 7)):

        if label_lst is None:
            label_lst = list(data.columns)

        if color_lst is None:
            n = len(label_lst)
            color_lst = [plt.get_cmap('viridis')(1. * i / n) for i in range(n)]

        if include_mu is not None:
            mu = int(np.ceil(np.mean(y)))

        if limit is not None:
            data = data[:limit]

        fig, ax = plt.subplots(figsize=fig_size)

        count = 0
        for ind in label_lst:
            y = data[ind]
            x_axis = range(len(y))
            if spacing is not None:
                x_axis = range(len([i for i in insert_every(x_axis, '', spacing)]))
                y = [i for i in insert_every(y, 0, spacing)]
            ax.bar(x_axis, y, align='center', color=color_lst[count], label=ind)
            count += 1

        if include_mu is not None:
            ax.plot(x_axis, [mu] * len(x), linestyle='--', color=mu_color, label='mu: ' + str(mu))

        plt.xticks(x_axis, x, rotation=xtick_rotation, fontsize=xtick_size)
        plt.ylabel(ylabel, fontsize=ylabel_size)
        plt.xlabel(xlabel, fontsize=xlabel_size)
        plt.title(title, fontsize=title_size)

        if grid:
            ax.grid(alpha=grid_alpha, linestyle=(0, grid_dash_sequence), linewidth=grid_lineweight)

        if include_mu is not None:
            plt.legend(fontsize=legend_fontsize, framealpha=legend_transparency, loc=legend_location, frameon=True)

        self.ax = ax

    def __repr__(self):
        return 'Bar Chart Plot'
