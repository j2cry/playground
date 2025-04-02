# For importing in notebooks
# import sys
# import os
# sys.path.append(os.path.abspath(os.pardir))

from __future__ import annotations
import re
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Self,
    Sequence,
    overload
)


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import scipy as sp
from tqdm import tqdm

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pandas._typing import (
        AggFuncTypeFrame,
        AstypeArg,
        Dtype
    )


class Select(TransformerMixin):
    @overload
    def __init__(self,
                 func: Literal[None],
                 /,
                 mode: Literal['drop', 'keep'] = 'keep'
                 ): ...

    @overload
    def __init__(self,
                 func: Callable[[str], bool],
                 /,
                 mode: Literal['drop', 'keep'] = 'keep'
                 ): ...

    @overload
    def __init__(self,
                 fields: list[str],
                 /,
                 mode: Literal['drop', 'keep'] = 'keep'
                 ): ...

    @overload
    def __init__(self,
                 pattern: str,
                 /,
                 mode: Literal['drop', 'keep'] = 'keep'
                 ): ...

    def __init__(
            self,
            selector: str | list[str] | Callable[[str], bool] | None,
            mode: Literal['drop', 'keep'] = 'keep'
    ):
        self.selector = selector
        self.mode = mode

    def _select_columns(self, X: pd.DataFrame) -> list[str]:
        if isinstance(self.selector, str):
            return [name for name in X.columns if re.match(self.selector, name)]
        elif callable(self.selector):
            return list(filter(self.selector, X.columns))
        elif isinstance(self.selector, (list, tuple, set)):
            return X.columns[X.columns.isin(self.selector)].to_list()
        else:
            return X.columns.to_list()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mask = X.columns.isin(self._select_columns(X))
        columns = X.columns[~mask] if self.mode == 'drop' else X.columns[mask]
        return X[columns].copy()


class WithSelected(Select):
    """Apply sub-pipeline for selected fields"""
    @overload
    def __init__(self, all: Literal[None], /, prefix: str = ''): ...
    @overload
    def __init__(self, func: Callable[[str], bool], /, prefix: str = ''): ...
    @overload
    def __init__(self, fields: list[str], /, prefix: str = ''): ...
    @overload
    def __init__(self, pattern: str, /, prefix: str = ''): ...

    def __init__(
            self,
            selector: str | list[str] | Callable[[str], bool] | None,
            prefix: str = '',
            propagate: bool = True
    ):
        super().__init__(selector, mode='keep')
        self.prefix = prefix
        self._steps = ()
        self.propagate = propagate

    def __call__(self, *steps: TransformerMixin) -> Self:
        self._steps = steps
        return self

    # @overload
    # def out(self, func: Callable[[str], bool], /) -> Self: ...
    # @overload
    # def out(self, fields: list[str], /) -> Self: ...
    # @overload
    # def out(self, pattern: str, /) -> Self: ...

    # def out(self, selector: str | list[str] | Callable[[str], bool]) -> Self:
    #     """Select output columns"""
    #     self._steps = (*self._steps, Select(selector, mode='keep'))
    #     return self

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        columns = self._select_columns(X)
        for step in self._steps:
            step.fit(X[columns], y)     # type: ignore
        return self

    def transform(self, X: pd.DataFrame, **fit_params) -> pd.DataFrame:
        X = X.copy()
        df = super().transform(X)
        selected = df.columns.to_list()
        for step in self._steps:
            df = step.transform(df if self.propagate else X[selected], **fit_params)  # type: ignore
        # resolve column names
        if len(selected) == 1 and not hasattr(df, 'columns'):
            columns = [f'{selected[0]}_{n}' for n in range(df.shape[1])]
            X[columns] = df.toarray()   # type: ignore
        else:
            columns = [f'{self.prefix}{name}' if self.prefix and name in selected else name
                       for name in df.columns]
            X.loc[:, columns] = df.values
        return X


class Apply(TransformerMixin):
    def __init__(
            self,
            estimator: BaseEstimator,
            locpipe: TransformerMixin | None = None,
            on: Iterable[str] | None = None,
            to: str = '',
            as_proba: bool = False
    ):
        self.estimator = estimator
        self.locpipe = locpipe
        self.on = on
        assert to, "Target feature name must be set"
        self.to = to
        self.as_proba = as_proba

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if self.on is None:
            self.on = X.columns
        df = self.locpipe.fit_transform(X[self.on], y) if self.locpipe is not None else X[self.on]
        self.estimator.fit(df, y)   # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        df = (self.locpipe.transform(X[self.on])    # type: ignore
              if self.locpipe is not None else X[self.on])
        if hasattr(self.estimator, 'predict_proba') and self.as_proba:
            X[self.to] = self.estimator.predict_proba(df).T[1]  # type: ignore
        elif hasattr(self.estimator, 'predict'):
            X[self.to] = self.estimator.predict(df)     # type: ignore
        elif hasattr(self.estimator, 'tramsform'):
            X[self.to] = self.estimator.transform(df)   # type: ignore
        else:
            classname = self.estimator.__class__.__name__
            raise AttributeError(f'{classname} do not implement any default prediction method')
        return X


class Calc(TransformerMixin):
    """Calculate new feature from existing"""
    def __init__(self, expr: str | Callable, to: str):
        self.expr = expr
        self.to = to
        self.trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if not callable(self.expr) and not self.trained:
            _expr = self.expr
            for name in set(re.findall(r'\b\w+\b', self.expr)):
                if name not in X.columns:
                    continue
                pattern = rf'\b{name}\b'
                _expr = re.sub(pattern, f'X["{name}"]', _expr)
            self.expr = _expr
            self.trained = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.to] = X.apply(self.expr, axis=1) if callable(self.expr) else eval(self.expr)
        return X


class Lag(TransformerMixin):
    """Calculate last N statistic"""
    def __init__(self, n: int = 1, stat: str | Callable[[Any], Any] = 'mean'):
        self.n = n
        self.stat = stat

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        shifted = []
        for step in range(self.n):
            shifted.append(X.shift(step + 1).fillna(X.iloc[0]))
        df = pd.concat(shifted)
        return df.groupby(df.index).agg(self.stat)


class Group(TransformerMixin):
    """Calculate new feature from Group by"""
    def __init__(
            self,
            by: str | list[str],
            aggregation: AggFuncTypeFrame,
            prefix: str = 'agg_',
            include_target: bool = False
    ):
        self.by = [by] if isinstance(by, str) else by
        self.aggregation = aggregation
        self._statistics = None
        self._prefix = prefix
        self._include_target = include_target

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if y is not None and self._include_target:
            assert y.name not in self.by, 'Grouping by target is logically incorrect'
            X = X.copy()
            X[y.name] = y
        self._statistics = X.groupby(self.by, as_index=False).agg(self.aggregation)
        mapper = {
            name: f'{self._prefix}{name}'
            for name in self._statistics.columns
            if name not in self.by
        }
        self._statistics.rename(columns=mapper, inplace=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self._statistics is not None:
            return X.merge(self._statistics, how='left')
        return X


class Fill(TransformerMixin):
    """Fill NaN with statistic or value"""
    @overload
    def __init__(self, obj: object, /): ...
    @overload
    def __init__(self, stat: Literal['mean', 'median'] = 'mean', /): ...

    def __init__(self, value: object | Literal['mean', 'median'] = 'mean', /):
        self.value = value
        self._statistics = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if self.value == 'mean':
            self._statistics = X.mean()
        elif self.value == 'median':
            self._statistics = X.median()
        elif re.search(r'[\'"](mean|median)[\'"]', str(self.value)):
            self._statistics = str(self.value).strip('"\'')
        else:
            self._statistics = self.value
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._statistics is not None, 'Fill block is not fit'
        return X.fillna(self._statistics)   # type: ignore


class Swap(TransformerMixin):
    """Swap values"""
    @overload
    def __init__(
        self,
        condition: Callable[[pd.DataFrame], pd.Series[bool]],
        left: str,
        right: str
    ): ...

    @overload
    def __init__(
        self,
        condition: Callable[[pd.DataFrame], pd.Series[bool]],
        left: list[str],
        right: list[str]
    ): ...

    def __init__(
            self,
            condition: Callable[[pd.DataFrame], pd.Series[bool]],
            left: str | list[str],
            right: str | list[str]
    ):
        assert isinstance(left, type(right)), 'left and right must be the same type'
        if isinstance(left, list):
            assert len(left) == len(right), 'left and right must be the same length'
        self.condition = condition
        self.left = left
        self.right = right

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        condition = self.condition(X)
        temp = X.loc[condition, self.left]
        X.loc[condition, self.left] = X.loc[condition, self.right]
        X.loc[condition, self.right] = temp
        return X


class Bins(TransformerMixin):
    """Make bins"""
    def __init__(
            self,
            on: str,
            to: str,
            bins: int | Sequence[int] | Sequence[float],
            as_: Literal['code', 'left', 'right'] = 'code',
    ):
        self.on = on
        self.to = to
        self.bins = bins
        self.categories = None
        self.as_ = as_

    def __get_bound(self, cat: pd.Interval):
        return cat.left if self.as_ == 'left' else cat.right

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.categories = pd.cut(X[self.on], bins=self.bins).cat.categories
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.categories is not None, 'Bins transformer is not trained'
        X = X.copy()
        values = pd.cut(X[self.on], bins=self.categories)
        if self.as_ == 'code':
            X[self.to] = values.cat.codes.astype(np.int64)
        else:
            X[self.to] = values.apply(self.__get_bound).astype(np.float64)
        return X


class KeepDataframe(TransformerMixin):
    """Keep result as DataFrame"""
    def __init__(self, transformer: TransformerMixin | Pipeline):
        self._transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._transformer.fit(X, y)     # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = self._transformer.transform(X)     # type: ignore
        return pd.DataFrame(result, index=X.index, columns=X.columns)


class TypeRecast(TransformerMixin):
    def __init__(self, dtype: AstypeArg | Mapping[Any, Dtype] | pd.Series):
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.astype(self.dtype, copy=True)


def boruta(X, y, estimator, iterations=20, alpha=0.05, seed=None, plot=False):
    np.random.seed(seed)
    hits = np.zeros(X.columns.size)
    for n in tqdm(range(iterations), total=iterations):
        # generate X_shadow
        X_shadow = X.apply(np.random.permutation)\
            .rename(columns={col: f'shadow_{col}' for col in X.columns})
        X_boruta = pd.concat([X, X_shadow], axis=1)
        # fit
        estimator.fit(X_boruta, y)
        # store importances
        imp_origin = estimator.feature_importances_[:X.columns.size]
        imp_shadow = estimator.feature_importances_[X.columns.size:]
        # calc hits
        hits += imp_origin > imp_shadow.max()
    # calc
    importances = pd.Series([sp.stats.binom.cdf(val, iterations, .5)
                             for val in hits], index=X.columns)
    if plot:
        importances[importances > (1 - alpha)]\
            .sort_values().plot(
                kind='barh',
                title=f'Boruta ft importances for {estimator.__class__.__name__}',
                color='goldenrod'
            )
    return importances
