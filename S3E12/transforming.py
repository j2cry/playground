import re
import numpy as np
import pandas as pd
import  sklearn.preprocessing as pre
from sklearn.base import BaseEstimator, TransformerMixin

import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm


class WithSelected:
    def __init__(self, fields, suffix):
        self.fields = fields
        self.suffix = suffix
        self.__steps = None

    def __call__(self, *steps):
        self.__steps = steps
        return self

    def fit(self, X, y=None):
        if callable(self.fields):
            self.fields = self.fields(X.columns)
        self.fields = X.columns[X.columns.isin(self.fields)] if self.fields is not None else X.columns
        for step in self.__steps:
            step.fit(X[self.fields], y)
        return self

    def transform(self, X, **fit_params):
        X = X.copy()
        for step in self.__steps:
            df = step.transform(X[self.fields], **fit_params)
            if df.shape[1] == len(self.fields):
                columns = [f'{field}_{self.suffix}' for field in self.fields]
            else:
                count = len(self.fields)
                columns = [f'{field}_{self.suffix}_{num}' for field in self.fields for num in range(df.shape[1] // len(self.fields))]
            X[columns] = df.toarray() if hasattr(df, 'toarray') else df
        return X


class Select(BaseEstimator, TransformerMixin):
    def __init__(self, fields, mode='keep'):
        assert mode in ('drop', 'keep'), "Given mode is not supported. Must be 'drop' or 'keep'"
        self.fields = fields
        self.mode = mode

    def fit(self, X, y=None):
        if callable(self.fields):
            self.fields = self.fields(X.columns)
        return self

    def transform(self, X):
        mask = X.columns.isin(self.fields)
        columns = X.columns[~mask] if self.mode == 'drop' else X.columns[mask]
        return X[columns]


class Apply(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, locpipe=None, on=None, to='', as_proba=False):
        self.estimator = estimator
        self.locpipe = locpipe
        self.on = on
        assert to, "Target feature name must be set"
        self.to = to
        self.as_proba = as_proba

    def fit(self, X, y=None):
        if self.on is None:
            self.on = X.columns
        elif callable(self.on):
            self.on = self.on(X.columns)
        tf = self.locpipe.fit_transform(X[self.on], y) if self.locpipe is not None else X[self.on]
        self.estimator.fit(tf, y)
        return self

    def transform(self, X):
        X = X.copy()
        tf = self.locpipe.transform(X[self.on]) if self.locpipe is not None else X[self.on]
        if hasattr(self.estimator, 'predict_proba') and self.as_proba:
            X[self.to] = self.estimator.predict_proba(tf).T[1]
        elif hasattr(self.estimator, 'predict'):
            X[self.to] = self.estimator.predict(tf)
        elif hasattr(self.estimator, 'tramsform'):
            X[self.to] = self.estimator.transform(tf)
        else:
            X[self.to] = np.nan
        return X


class Calc(BaseEstimator, TransformerMixin):
    def __init__(self, expr, to):
        self.expr = expr
        self._expr = None
        self.to = to

    def fit(self, X, y=None):
        fields = re.findall('\w+', self.expr)
        self._expr = self.expr
        replaced = []
        for f in fields:
            if f in X.columns and f not in replaced:
                self._expr = self._expr.replace(f, f'X["{f}"]')
                replaced.append(f)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.to] = eval(self._expr)
        return X


class TypeRecast(BaseEstimator, TransformerMixin):
    def __init__(self, **mapping):
        self.mapping = mapping
    
    def fit(self, X, y=None):
        self.mapping = {col: t for t, f in self.mapping.items() for col in (f(X.columns) if callable(f) else f)}
        return self

    def transform(self, X):
        return X.astype(self.mapping)


def boruta(X, y, estimator, iterations=20, alpha=0.05, seed=None, plot=False):
    np.random.seed(seed)
    hits = np.zeros(X.columns.size)
    for n in tqdm(range(iterations), total=iterations):
        # generate X_shadow
        X_shadow = X.apply(np.random.permutation).rename(columns={col: f'shadow_{col}' for col in X.columns})
        X_boruta = pd.concat([X, X_shadow], axis=1)
        # fit
        estimator.fit(X_boruta, y)
        # store importances
        imp_origin = estimator.feature_importances_[:X.columns.size]
        imp_shadow = estimator.feature_importances_[X.columns.size:]
        # calc hits
        hits += imp_origin > imp_shadow.max()
    # calc
    importances = pd.Series([sp.stats.binom.cdf(val, iterations, .5) for val in hits], index=X.columns)
    if plot:
        importances[importances > (1 - alpha)].sort_values().plot(kind='barh', title=f'Boruta ft importances for {estimator.__class__.__name__}', color='goldenrod');
    return importances

