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
