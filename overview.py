# For importing in notebooks
# import sys
# import os
# sys.path.append(os.path.abspath(os.pardir))

import enum
import pandas as pd
import scipy.stats as stats
from matplotlib.axes import Axes


class FigureType(enum.IntEnum):
    HIST = enum.auto()
    BAR = enum.auto()
    DENSITY = enum.auto()


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
            container = ax.hist(feature, **kw)
            if kw.get('density'):
                density = stats.gaussian_kde(feature)
                ax.plot(container[1], density(container[1]), c=kw.get('color'))
        case FigureType.BAR:
            ax.bar(feature.index, feature, **kw)
            ax.set_xticks(feature.index, feature.index, rotation=rotation)
        case FigureType.DENSITY:
            density = stats.gaussian_kde(feature)
            _, xrange = pd.cut(feature, bins=kw.pop('bins', 10), retbins=True)
            ax.plot(xrange, density(xrange), **kw)
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
