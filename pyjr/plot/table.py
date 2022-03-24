"""
Table plot class.

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
class Table:
    """

    Class for plotting tables.

    :param data: Input data.
    :type data: pd.DataFrame
    :param label_lst: List of labels to include, if None will include all columns. *Optional*
    :type label_lst: List[str]
    :param fig_size: default = (10, 10), *Optional*
    :type fig_size: tuple
    :param font_size: Font size inside cells, default = 'medium'. *Optional*
    :type font_size: str
    :param font_color: Color of text inside cells, default is 'black'. *Optional*
    :type font_color: str
    :param col_widths: Width of columns, default = 0.30. *Optional*
    :type col_widths: float
    :param row_colors: Color of rows. *Optional*
    :type row_colors: str
    :param header_colors: Header of table color. *Optional*
    :type header_colors: str
    :param edge_color: Color of cell edges, default = 'w'. *Optional*
    :type edge_color: str
    :param sequential_cells: If True will color ever other row. *Optional*
    :type sequential_cells: bool
    :param color_map: Color map used in cells, default = 'Greens'. *Optional*
    :type color_map: str
    :example: *None*
    :note: *None*

    """

    __slots__ = ("ax")

    # import matplotlib
    # matplotlib.colors.to_rgba(c, alpha=None)
    def __init__(self,
                 data: pd.DataFrame,
                 label_lst: Optional[List[str]] = None,
                 fig_size: Optional[tuple] = (10, 10),
                 font_size: Optional[str] = 'medium',
                 col_widths: Optional[float] = 0.30,
                 row_colors: Optional[str] = None,
                 header_colors: Optional[str] = None,
                 edge_color: Optional[str] = 'w',
                 sequential_cells: Optional[bool] = None,
                 color_map: Optional[str] = 'Greens',
                 font_color: Optional[str] = 'black',
                 ):
        data['index'] = list(data.index)

        if row_colors is None:
            row_colors = ['#f1f1f2', 'w']
        if type(row_colors) is str:
            row_colors = [row_colors, 'w']
        if header_colors is None:
            header_colors = ['tab:blue', 'w']
        if type(header_colors) is str:
            header_colors = [header_colors, 'w']

        if label_lst is None:
            lst = list(data.columns)
            lst.remove('index')
            label_lst = ['index'] + lst
        data = data[label_lst]
        col_widths = [col_widths] * len(label_lst)
        colours = None
        if sequential_cells is not None:
            color_lst = []
            for col in label_lst:
                if type(data[col].iloc[0]) != str and col != 'index':
                    _norm = plt.Normalize(np.min(data[col]) - 1, np.max(data[col]) + 1)
                    temp = plt.get_cmap(color_map)(_norm(data[col]))
                elif type(data[col].iloc[0]) == str and col != 'index':
                    temp = [(1.0, 1.0, 1.0, 1.0), (0.945, 0.945, 0.949, 1.0)] * len(data)
                else:
                    temp = [(0.121, 0.466, 0.705, 0.15), (0.121, 0.466, 0.705, 0.30)] * len(data)
                temp_lst = []
                for i in range(len(data)):
                    temp_lst.append(tuple(temp[i]))
                color_lst.append(temp_lst)
            colours = np.array(pd.DataFrame(color_lst).T)

        fig, ax = plt.subplots(figsize=fig_size)
        table = ax.table(cellText=data.values, colLabels=label_lst, colWidths=col_widths, loc='center',
                         cellLoc='center', cellColours=colours)
        table.set_fontsize(font_size)

        for k, cell in six.iteritems(table._cells):
            r, c = k
            cell.set_edgecolor(edge_color)
            if r == 0:
                cell.set_text_props(weight='bold', color=header_colors[1])
                cell.set_facecolor(header_colors[0])
            else:
                if sequential_cells is None:
                    cell.set_facecolor(row_colors[r % len(row_colors)])
                if c != 0 and c != 'index':
                    cell.set_text_props(color=font_color)

        ax.axis('tight')
        ax.axis('off')
        fig.tight_layout()

        self.ax = ax

    def __repr__(self):
        return 'Table Plot'
