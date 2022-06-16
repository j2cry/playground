import pathlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from collections import namedtuple
from typing import Mapping, Iterable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import parallel_backend
from joblib import Parallel, delayed

Step = namedtuple('Step', 'callable,columns,parameters')


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


class ImputeHelper():
    def __init__(self, *steps):
        self.steps = steps

    def run(self, data, inherit=True):
        """
        :params data - original dataset
        :params inherit - put the previous result to the next step input
        """
        predicted = data.copy()
        for step in map(Step, *zip(*self.steps)):
            columns = step.columns if step.columns != 'all' else data.columns
            df = data[columns] if not inherit else predicted[columns]
            # call step function
            if isinstance(step.parameters, Mapping):
                step_result = step.callable(df, **step.parameters)
            elif isinstance(step.parameters, Iterable) and not isinstance(step.parameters, str):
                step_result = step.callable(df, *step.parameters)
            else:
                step_result = step.callable(df, step.parameters)
            predicted.fillna(step_result[columns], inplace=True)
        return predicted


class CosineSimilarity:
    def __init__(self, df):
        self.use_cols = df.columns[~df.isna().any()]
        self.nan_rows = df.isna().any(axis=1)
        self.df = df
    
    def _process_chunk(self, chunk, k, threshold):
        cosine = cosine_similarity(chunk, self.df[self.use_cols])
        if threshold is not None:       # collect by threshold
            mask = cosine > threshold       # apply threshold
            np.fill_diagonal(mask, False)   # exclude ownes
            # return list(map(lambda m: self.df[m].mean().values, mask))
            return np.apply_along_axis(lambda m: self.df[m].mean(), 1, mask)
        elif k is not None:     # collect by k nearest
            # return list(map(lambda c: self.df.iloc[np.argsort(c)[::-1][1:]].head(k).mean().values, cosine))
            return np.apply_along_axis(lambda c: self.df.iloc[np.argsort(c)[::-1][1:]].head(k).mean(), 1, cosine)

    def calculate(self, *, k=None, threshold=None, chunksize=100, backend='loky'):
        assert bool(k) ^ bool(threshold), 'Only one parameter must be specified: either `k` or `threshold`.'

        df_size = self.df[self.nan_rows].index.size
        chunk_count = df_size // chunksize + (1 if df_size % chunksize else 0)
        print(f'Chunks: {chunk_count}')
        arr = []
        # for start in tqdm(range(0, df_size, chunksize), total=chunk_count):
        #     arr.extend(self._process_chunk(self.df.loc[self.nan_rows, self.use_cols].iloc[start:start + chunksize], k, threshold))
        if backend is not None:
            with parallel_backend(backend):
                result = (Parallel(n_jobs=-1)(
                    delayed(self._process_chunk)(self.df.loc[self.nan_rows, self.use_cols].iloc[start:start + chunksize], k, threshold) 
                        for start in tqdm(range(0, df_size, chunksize), total=chunk_count)
                ))
        else:
            result = [self._process_chunk(self.df.loc[self.nan_rows, self.use_cols].iloc[start:start + chunksize], k, threshold)
                        for start in tqdm(range(0, df_size, chunksize), total=chunk_count)]
        # parse result
        for part in result:
            arr.extend(part)
        stats = pd.DataFrame(arr, index=self.df[self.nan_rows].index)
        if stats.isna().any().any():
            print('Some values are NaN. Try decrease threshold.')
        return stats


def calc_score(df, pred):
    # compute score
    metrics = []
    nan_cols = df.columns[df.isna().any()]
    for col in tqdm(nan_cols):
        nan_rows = df[col].isna()   # select rows that are NaN in this columns
        y_train = df.loc[~nan_rows, col]
        train_pred = pred.loc[~nan_rows, col]
        metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))
    return np.mean(metrics)


def ml_impute(df, estimator, *, max_fill_nan_count=np.inf, max_train_nan_count=np.inf):
    """ Impute data with one-per-column estimators
    :param df - original dataset
    :param estimator - model to be fit
    :param max_fill_nan_count - max allowed number of NaN in row to predict value in it (before train/test split)
    :param max_train_nan_count - flag to use data containing NaN in train. If True, NaN will be filled in with mean.
    """
    assert max_fill_nan_count >= 1, "`max_fill_nan_count` < 1 doesn't make sense. No value will be predicted."
    assert max_train_nan_count >= 0, "`max_train_nan_count` < 0 doesn't make sense. No data left for training."

    pred = df.copy()
    metrics = []
    nan_cols = df.columns[df.isna().any()]      # select columns with nans
    fill_nan_count = df.isna().sum(axis=1) <= max_fill_nan_count
    train_nan_count = df.isna().any(axis=1) <= max_train_nan_count

    for col in tqdm(nan_cols):
        target_mask = df[col].isna()   # select rows that are NaN in this columns
        # train/test split
        X_train = df[~target_mask & train_nan_count].drop(col, axis=1)
        y_train = df.loc[~target_mask & train_nan_count, col]
        X_test = df[target_mask & fill_nan_count].drop(col, axis=1)
        
        # fit/predict
        estimator.fit(X_train, y_train)
        train_pred = estimator.predict(X_train)
        pred.loc[target_mask & fill_nan_count, col] = estimator.predict(X_test)
        # score pipeline
        metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))    # RMSE
    print(f'\nML Imputer avg. score: {np.mean(metrics)}')
    return pred


def simplestat(df, imputer):
    """ Impute data with SimpleImputer """
    nan_cols = df.columns[df.isna().any()]      # get columns with nans
    # fit/predict
    imputer.fit(df[nan_cols])
    values = np.full(df[nan_cols].shape, imputer[-1].statistics_ if isinstance(imputer, Pipeline) else imputer.statistics_)
    stats = pd.DataFrame(values, index=df.index, columns=df[nan_cols].columns)
    pred = df.fillna(stats)
    print(f'SimpleStat avg. score: {calc_score(df, stats)}')
    return pred


def groupstat(df, *, gcol, func='mean'):
    """ Impute data with grouped statistics """
    stats = df.groupby(gcol).transform(func)
    if stats.isna().any().any():
        stats.fillna(stats.agg(func), inplace=True)
        print(f'Stats contain NaNs! Data may not be filled in completely!')
    pred = df.fillna(stats)
    print(f'GroupStat avg. score: {calc_score(df, stats)}')
    return pred


def cosine_stats(df, *, k=None, threshold=None, backend='loky', chunksize=100):
    csim = CosineSimilarity(df)
    stats = csim.calculate(k=k, threshold=threshold, backend=backend, chunksize=chunksize)
    pred = df.fillna(stats)
    print(f'SimpleStat avg. score: {calc_score(df, stats)}')
    return pred
