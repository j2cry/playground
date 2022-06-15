import pathlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from collections import namedtuple
from typing import Mapping, Iterable


Step = namedtuple('Step', 'callable,columns,parameters')


path = pathlib.Path().joinpath('data')
data_path = path.joinpath('data.csv')

# read data
data = pd.read_csv(data_path, index_col='row_id')
# subset split
n_sub = 4
subset = [data.columns, ]
for n in range(n_sub):
    columns = data.columns[data.columns.str.startswith(f'F_{n + 1}')]
    subset.append(columns)


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


def ml_impute(df, estimator):
    """ Impute data with one-per-column estimators """
    pred = df.copy()
    metrics = []
    nan_cols = df.columns[df.isna().any()]      # get columns with nans
    for col in tqdm(nan_cols):
        nan_rows = df[col].isna()   # select rows that are NaN in this columns
        # train/test split
        X_train = df[~nan_rows].drop(col, axis=1)       # fit on WHOLE data TODO: not_na_train parameter
        y_train = df.loc[~nan_rows, col]
        X_test = df[nan_rows].drop(col, axis=1)
        
        # fit/predict
        estimator.fit(X_train, y_train)
        train_pred = estimator.predict(X_train)
        pred.loc[nan_rows, col] = estimator.predict(X_test)
        # score pipeline
        metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))    # RMSE
    print(f'\nML Imputer avg. score: {np.mean(metrics)}')
    return pred


def simplestat(df, imputer):
    """ Impute data with SimpleImputer """
    nan_cols = df.columns[df.isna().any()]      # get columns with nans
    # fit/predict
    imputer.fit(df[nan_cols])
    values = np.full(df.shape, imputer[-1].statistics_ if isinstance(imputer, Pipeline) else imputer.statistics_)
    stats = pd.DataFrame(values, index=df.index, columns=df.columns)    
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
