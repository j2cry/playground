import numpy as np
import pandas as pd
import  sklearn.preprocessing as pre
from sklearn.base import BaseEstimator, TransformerMixin


class WithSelected:
    def __init__(self, fields, suffix):
        self.fields = fields
        self.suffix = suffix
        self.__steps = None
    
    def __call__(self, *steps):
        self.__steps = steps
        return self
    
    def fit(self, X, y=None):
        self.fields = X.columns[X.columns.isin(self.fields)]
        for step in self.__steps:
            step.fit(X[self.fields], y)
        return self
    
    def transform(self, X, **fit_params):
        X = X.copy()
        for step in self.__steps:
            df = step.transform(X[self.fields], **fit_params)
            X[[f'{field}_{self.suffix}' for field in self.fields]] = df
        return X


class DFPowerTransform(pre.PowerTransformer):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), index=X.index, columns=X.columns)

    def fit_transform(self, X, y=None):
        return pd.DataFrame(super().fit_transform(X, y), index=X.index, columns=X.columns)


class Drop(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns = X.columns[~X.columns.isin(self.fields)]
        return X[columns]


class Apply(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, locpipe=None, on=None, to=''):
        self.estimator = estimator
        self.locpipe = locpipe
        self.on = on
        assert to, "Target feature name must be set"
        self.to = to
    
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
        if hasattr(self.estimator, 'predict'):
            X[self.to] = self.estimator.predict(tf)
        elif hasattr(self.estimator, 'tramsform'):
            X[self.to] = self.estimator.transform(tf)
        else:
            X[self.to] = np.nan
        return X
