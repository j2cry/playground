# For importing in notebooks
# import sys
# import os
# sys.path.append(os.path.abspath(os.pardir))

import enum
import pandas as pd
from matplotlib.axes import Axes


class FigureType(enum.IntEnum):
    HIST = enum.auto()
    BAR = enum.auto()


class StatisticLine(enum.IntFlag):
    NONE = 0
    ZERO = enum.auto()
    MEAN = enum.auto()
    MEDIAN = enum.auto()
    MODE = enum.auto()
    BASIC = MEAN | MEDIAN
    ALL = MEAN | MEDIAN | MODE


def feature_overview(ax: Axes,
                     feature: pd.Series,
                     ftype: FigureType = FigureType.HIST,
                     statline: StatisticLine = StatisticLine.NONE,
                     **kw
                     ):
    ax.set_title(f'Distribution of `{feature.name}`')
    ax.set_xlabel(str(feature.name))
    ax.set_ylabel('Density')
    # plot distribution
    rotation = kw.pop('r', 0)
    match ftype:
        case FigureType.HIST:
            ax.hist(feature, **kw)
        case FigureType.BAR:
            counts = feature.value_counts()
            ax.bar(counts.index, counts, **kw)
            ax.set_xticks(counts.index, counts.index, rotation=rotation)
    # plot statistic lines
    if statline:
        if StatisticLine.ZERO in statline:
            ax.axvline(0, linewidth=1, c='blue', ls='-', lw=1, label='zero')
        if StatisticLine.MEAN in statline:
            ax.axvline(feature.mean(), c='orange', ls='-', lw=1, label='mean')
        if StatisticLine.MEDIAN in statline:
            ax.axvline(feature.median(), c='green', ls='-', lw=1, label='median')
        if StatisticLine.MODE in statline:
            modes = feature.mode()
            color_step = int(255 / modes.size)
            for n, mode in enumerate(modes):
                color = f'#{hex(255 - color_step * n)[2:]}0000'
                ax.axvline(mode, c=color, linewidth=1, label=f'mode {n + 1}')
        ax.legend()
