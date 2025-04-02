from __future__ import annotations
import datetime as dt
import random
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Feature converters
def temparature(x: pd.Series) -> float:
    """Fix `temparature` features"""
    if x.mintemp <= x.temparature <= x.maxtemp:
        return x.temparature
    else:
        return sorted([x.mintemp, x.temparature, x.maxtemp])[1]


def season(s: pd.Series) -> int:
    """Create `season` feature"""
    if 60 <= s.day <= 151:
        return 2    # spring
    elif 152 <= s.day <= 243:
        return 3    # summer
    elif 244 <= s.day <= 334:
        return 4    # autumn
    else:
        return 1    # winter


def month(s: pd.Series) -> int:
    """Create `month` feature"""
    base = dt.date(1, 1, 1)
    return (base + dt.timedelta(days=s.day - 1)).month


# -----------------------------------------------------------------------------
# Ensemble
class EnsembleEstimator[T](Protocol):
    n_estimators: int
    # estimator_class: type[T]
    models: tuple[T, ...]
    random_state: int | None

    def fit(self, X, y, **fit_params): ...
    def predict(self, X): ...
    def predict_proba(self, X, klass: int | None = None): ...


class EnsembleMeta[**P, R](type):
    def __new__(
        cls,
        estimator_class: Callable[P, R],
        n_estimators: int = 50,
        random_state: int | None = None,
        random_state_name: str | None = 'random_state',
    ) -> Callable[P, EnsembleEstimator[R]]:
        classname = f'Ensemble{estimator_class.__name__}'
        namespace = {
            '__new__': cls._new,
            'estimator_class': estimator_class,
            'n_estimators': n_estimators,
            'random_state': random_state,
            'random_state_name': random_state_name,
            'fit': cls._fit,
            'predict': cls._predict,
            'predict_proba': cls._predict_proba,
        }
        return type(classname, (), namespace)     # type: ignore
        # return super().__new__(cls, classname, (BaseEstimator,), namespace)

    @staticmethod
    def _new(_class, *args: P.args, **kwargs: P.kwargs) -> ...:
        instance = super(_class, _class).__new__(_class)
        if isinstance(rs := kwargs.pop('random_state', None), (int, type(None))):
            instance.random_state = rs
        else:
            instance.random_state = None
        random.seed(instance.random_state)
        # init models
        models = []
        for _ in range(instance.n_estimators):
            if instance.random_state_name and instance.random_state:
                kwargs[instance.random_state_name] = random.randint(1, 2 ** 31)
            models.append(instance.estimator_class(*args, **kwargs))
        instance.models = tuple(models)
        return instance

    @staticmethod
    def _fit[T: EnsembleEstimator](instance: T, X, y, verbose=False, **fit_params) -> T:
        _models = tqdm(instance.models) if verbose else instance.models
        for model in _models:
            model.fit(X, y, **fit_params)
        return instance

    @staticmethod
    def _predict_proba(instance, X, verbose: bool = False):
        _models = tqdm(instance.models) if verbose else instance.models
        return np.mean([model.predict_proba(X) for model in _models], axis=0)

    @staticmethod
    def _predict(instance, X, verbose: bool = False):
        _models = tqdm(instance.models) if verbose else instance.models
        return np.mean([model.predict(X) for model in _models], axis=0)
