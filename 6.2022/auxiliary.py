import os
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import wraps

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from typing import Mapping, Iterable

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import parallel_backend
from joblib import Parallel, delayed


path = pathlib.Path().joinpath('data')
data_path = path.joinpath('data.csv')

# read data
data = pd.read_csv(data_path, index_col='row_id')
# subset split
n_sub = 4
subset = [data, ]
subcol = [data.columns, ]
for n in range(n_sub):
    columns = data.columns[data.columns.str.startswith(f'F_{n + 1}')].tolist()
    subset.append(data[columns])
    subcol.append(columns)    


def save_submission(predicted):
    print(f'Still contain NaN: {predicted.isna().any().any()}')
    sub_path = path.joinpath('sample_submission.csv')
    # collect predictions
    sub = pd.read_csv(sub_path)
    predict = sub['row-col'].str.split('-').apply(lambda bundle: predicted.loc[int(bundle[0]), bundle[1]])
    sub['value'] = predict
    sub.to_csv('submission.csv', index=False)
    return sub.head()


def deprecated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Using `{func.__name__}()` is deprecated! It may take a VERY long time on large data, work unstable or don't work at all.")
        return func(*args, **kwargs)
    return wrapper

# =============================== Impute Helper Definition ===============================
class Step:
    def __init__(self, func, columns, **kwargs):
        """
        :param func - step function, must have signature like func(train, test, **params)
        :param columns - columns passed to function
        :keyword max_fill_nan_count - max allowed number of NaN in row to predict value in it
        :keywird max_train_nan_count - max allowed number of NaN in row to use it in train. NaN will be filled in with mean
        """

        assert func is not callable, "`func` parameter must be callable"
        self.func = func
        self.columns = columns
        # parse parameters
        self.max_fill_nan_count = kwargs.pop('max_fill_nan_count', np.inf)
        assert self.max_fill_nan_count >= 1, "`max_fill_nan_count` < 1 doesn't make sense. No value will be predicted."
        self.max_train_nan_count = kwargs.pop('max_train_nan_count', np.inf)
        assert self.max_train_nan_count >= 0, "`max_train_nan_count` < 0 doesn't make sense. No data left for training."
        self.params = kwargs

    def call(self, df):
        # train/test split
        self.columns = self.columns if self.columns != 'all' else df.columns
        fill_nan_count = df.isna().sum(axis=1) <= self.max_fill_nan_count
        train_nan_count = df.isna().sum(axis=1) <= self.max_train_nan_count        
        train = df.loc[train_nan_count, self.columns]
        test = df.loc[fill_nan_count, self.columns]
        return self.func(train, test, **self.params)


class ImputeHelper():
    def __init__(self, *steps):
        self.steps = steps

    def run(self, data, inherit=True):
        """
        :params data - original dataset
        :params inherit - put the previous result to the next step input
        """
        predicted = data.copy()
        for step in self.steps:            
            # call step function
            step_result = step.call(predicted if inherit else data)
            predicted.fillna(step_result, inplace=True)
        return predicted


def calc_train_score(df, pred):
    # compute score
    metrics = []
    nan_cols = df.columns[df.isna().any()]
    for col in tqdm(nan_cols):
        nan_rows = df[col].isna()   # select rows that are NaN in this columns
        y_train = df.loc[~nan_rows, col]
        train_pred = pred.loc[~nan_rows, col]
        metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))
    return np.mean(metrics)


# =============================== Imputer step functions ===============================
def simplestat(train, test, imputer=SimpleImputer()):
    """ Impute data with SimpleImputer """
    imputer.fit(train)
    check = pd.DataFrame(imputer.transform(train), index=train.index, columns=train.columns)
    print(f'Simple imputer avg. score: {calc_train_score(train, check)}')
    values = pd.DataFrame(imputer.transform(test), index=test.index, columns=test.columns)
    pred = test.fillna(values)
    return 


def groupstat(train, test, *, gcol, func='mean'):
    """ Impute data with grouped statistics """
    stats = train.groupby(gcol).transform(func)
    if stats.isna().any().any():
        print(f'Stats contain NaN! Data may not filled in completely!')
    print(f'Groupstat imputer avg. score: {calc_train_score(train.drop(gcol, axis=1), stats)}')
    return test.fillna(stats)


def predictor(train, test, *, estimator):
    """ Impute data with one-per-column estimators """
    pred = test.copy()
    metrics = []
    nan_cols = test.columns[test.isna().any()]      # select columns to be filled in

    for col in tqdm(nan_cols):
        nan_target = train[col].isna()      # exclude rows without target from train
        target_mask = test[col].isna()      # select rows that are NaN in this columns
        # train/test split
        X_train = train[~nan_target].drop(col, axis=1)
        y_train = train.loc[~nan_target, col]
        X_test = test[target_mask].drop(col, axis=1)
        
        # fit/predict
        estimator.fit(X_train, y_train)
        train_pred = estimator.predict(X_train)
        pred.loc[target_mask, col] = estimator.predict(X_test)
        # score pipeline
        metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))    # RMSE
    print(f'\nML Imputer avg. score: {np.mean(metrics)}')
    return pred






# this takes a VERY long time
class CosineSimilarity:
    def __init__(self, train, test):
        nan_rows = test.isna().any(axis=1)
        self.use_cols = train.columns[~train.isna().any() & ~test.isna().any()]
        self.train = train
        self.test = test[nan_rows]
    
    def _process_chunk(self, chunk, k, threshold):
        cosine = cosine_similarity(chunk, self.train[self.use_cols])
        if threshold is not None:           # collect by threshold
            mask = cosine > threshold       # apply threshold
            np.fill_diagonal(mask, False)   # exclude ownes
            # return list(map(lambda m: self.df[m].mean().values, mask))
            return np.apply_along_axis(lambda m: self.train[m].mean(), 1, mask)
        elif k is not None:     # collect by k nearest
            # return list(map(lambda c: self.df.iloc[np.argsort(c)[::-1][1:]].head(k).mean().values, cosine))
            return np.apply_along_axis(lambda c: self.train.iloc[np.argsort(c)[::-1][1:]].head(k).mean(), 1, cosine)

    def calculate(self, *, k=None, threshold=None, backend=None, chunksize=50):
        assert bool(k) ^ bool(threshold), 'Only one parameter must be specified: either `k` or `threshold`.'

        test_size = self.test.index.size
        chunk_count = test_size // chunksize + (1 if test_size % chunksize else 0)
        print(f'{test_size} rows in {chunk_count} chunks')
        arr = []
        if backend is not None:
            with parallel_backend(backend):
                result = (Parallel(n_jobs=-1)(
                    delayed(self._process_chunk)(self.test[self.use_cols].iloc[start:start + chunksize], k, threshold) 
                        for start in tqdm(range(0, test_size, chunksize), total=chunk_count)
                ))
        else:
            result = [self._process_chunk(self.test[self.use_cols].iloc[start:start + chunksize], k, threshold)
                        for start in tqdm(range(0, test_size, chunksize), total=chunk_count)]
        # parse result
        for part in result:
            arr.extend(part)
        stats = pd.DataFrame(arr, index=self.test.index, columns=self.test.columns)
        if stats.isna().any().any():
            print('Some values are NaN. Try decrease threshold.')
        return stats


@deprecated
def cosine_stats(train, test, *, k=None, threshold=None, backend=None, chunksize=50):
    if backend is not None:
        os.environ['MKL_NUM_THREADS'] = '1'
    csim = CosineSimilarity(train, test)
    stats = csim.calculate(k=k, threshold=threshold, backend=backend, chunksize=chunksize)
    pred = test.fillna(stats)
    return pred
