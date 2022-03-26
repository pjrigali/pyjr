"""
Table plot class.

Usage:
 ./plot/table.py

Author:
 Peter Rigali - 2022-03-10
"""
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import six
from pyjr.classes.data import Data
from pyjr.classes.preprocess_data import PreProcess
from pyjr.utils.tools import _to_metatype
from pyjr.utils.base import _min, _max


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
                 data: Union[pd.DataFrame, Data, PreProcess, List[Union[Data, PreProcess]]],
                 label_lst: Optional[Union[List[str], Tuple[str]]] = None,
                 limit: Optional[Union[List[int], Tuple[int]]] = None,
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
        # Parse input data
        if isinstance(data, (Data, PreProcess)):
            # if label_lst is None:
            #     label_lst = _to_metatype(data=data.name, dtype='list')
            data = data.dataframe()
        elif isinstance(data, pd.DataFrame):
            pass
            # if label_lst is None:
                # label_lst = _to_metatype(data=data.columns, dtype='list')
        elif isinstance(data, list):
            dic = {}
            for d in data:
                if isinstance(d.name, (list, tuple)):
                    for ind, val in enumerate(d.name):
                        dic[val.name] = val.data[:, ind]
                else:
                    dic[d.name] = d.data
            data = pd.DataFrame.from_dict(dic)
            # label_lst = _to_metatype(data=data.columns, dtype='list')
        data['index'] = _to_metatype(data=data.index, dtype='list')

        if limit:
            data = data[limit[0]:limit[1]]

        if row_colors is None:
            row_colors = ['#f1f1f2', 'w']
        if type(row_colors) is str:
            row_colors = [row_colors, 'w']
        if header_colors is None:
            header_colors = ['tab:blue', 'w']
        if type(header_colors) is str:
            header_colors = [header_colors, 'w']

        # lst = _to_metatype(data=data.columns, dtype='list')
        # label_lst.remove('index')
        # label_lst = ['index'] + label_lst

        if label_lst is None:
            lst = _to_metatype(data=data.columns, dtype='list')
            lst.remove('index')
            label_lst = ['index'] + lst
        # data = data[label_lst]

        col_widths = [col_widths] * label_lst.__len__()
        colours = None
        if sequential_cells is not None:
            color_lst = []
            for col in label_lst:
                # if isinstance(data[col].iloc[0], str) is False and col != 'index':
                if type(data[col].iloc[0]) != str and col != 'index':
                    d = _to_metatype(data=data[col], dtype='list')
                    # _norm = plt.Normalize(np.min(data[col]) - 1, np.max(data[col]) + 1)
                    _norm = plt.Normalize(_min(data=d) - 1, _max(data=d) + 1)
                    temp = plt.get_cmap(color_map)(_norm(data[col]))
                # elif isinstance(data[col].iloc[0], str) and col != 'index':
                elif type(data[col].iloc[0]) == str and col != 'index':
                    temp = [(1.0, 1.0, 1.0, 1.0), (0.945, 0.945, 0.949, 1.0)] * data.__len__()
                else:
                    temp = [(0.121, 0.466, 0.705, 0.15), (0.121, 0.466, 0.705, 0.30)] * data.__len__()
                temp_lst = []
                for i in range(data.__len__()):
                    temp_lst.append(tuple(temp[i]))
                color_lst.append(temp_lst)
            colours = np.array(pd.DataFrame(color_lst).T)

        # Start plot
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
