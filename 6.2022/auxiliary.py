import pathlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from collections import namedtuple


Step = namedtuple('Step', 'callable,parameters')


path = pathlib.Path().joinpath('data')
data_path = path.joinpath('data.csv')
sub_path = path.joinpath('sample_submission.csv')

# read data
data = pd.read_csv(data_path, index_col='row_id')

# subset split
n_sub = 4
subset = [None, ]       # this is for a more familiar numbering
for n in range(n_sub):
    sub_cols = data.columns[data.columns.str.startswith(f'F_{n + 1}')]
    subset.append(data[sub_cols])


def save_submission(predicted):
    print(f'Still contain NaN: {predicted.isna().any().any()}')
    # collect predictions
    sub = pd.read_csv(sub_path)
    predict = sub['row-col'].str.split('-').apply(lambda bundle: predicted.loc[int(bundle[0]), bundle[1]])
    sub['value'] = predict
    sub.to_csv('submission.csv', index=False)
    return sub.head()


class ImputerHelper():
    def __init__(self, *steps):
        self.steps = steps

    def run(self, data, inherit=True):
        """
        :params data - original dataset
        :params inherit - put the previous result to the next step input
        """
        predicted = data.copy()
        for step in map(Step, *zip(*self.steps)):
            step_result = step.callable(data if not inherit else predicted, step.parameters)
            predicted.fillna(step_result, inplace=True)
        return predicted


def ml_impute(df, pipe):
    """
    :param pipe - estimator or transformer pipeline
    """
    imp = df.copy()
    metrics = []
    # get columns with nans
    nan_cols = df.columns[df.isna().any()]
    is_estimator = hasattr(pipe, 'predict')
    if not is_estimator:
        pipe.fit(df[nan_cols])    # fit if pipe is transformer
        # predict
        stats = pd.DataFrame([pipe[-1].statistics_ if isinstance(pipe, Pipeline) else pipe.statistics_] * df.shape[0], index=df.index, columns=df.columns)

    for col in tqdm(nan_cols):
        nan_rows = df[col].isna()   # select rows that are NaN in this columns
        y_train = df.loc[~nan_rows, col]

        # fit if pipe is estimator
        if is_estimator:
            # train/test split
            X_train = df[~nan_rows].drop(col, axis=1)
            X_test = df[nan_rows].drop(col, axis=1)
            pipe.fit(X_train, y_train)
            # predict
            train_pred = pipe.predict(X_train)
            imp.loc[nan_rows, col] = pipe.predict(X_test)
        else:
            # predict if pipe is transformer
            train_pred = stats.loc[~nan_rows, col]
            imp.loc[nan_rows, col] = stats.loc[nan_rows, col].values
          
        # score pipeline
        score = np.sqrt(mean_squared_error(y_train, train_pred))    # RMSE
    metrics.append(score)
    print(f'\n{np.mean(metrics)}')
    return imp


def groupstat(df, *, gcol, func='mean'):
    stats = df.groupby(gcol).transform(func)
    if stats.isna().any().any():
        stats.fillna(stats.agg(func), inplace=True)
        print(f'Stats contain NaNs! Filled with {func}')
    imp = df.fillna(stats)
    # if imp.isna().any().any():
    #     imp.fillna(imp.agg(func), inplace=True)
    #     print('Data still contain NaNs! Select another group columns')

    # compute score
    metrics = []
    nan_cols = df.columns[df.isna().any()]
    for col in tqdm(nan_cols):
        nan_rows = df[col].isna()   # select rows that are NaN in this columns
        y_train = df.loc[~nan_rows, col]
        train_pred = stats.loc[~nan_rows, col]
        score = np.sqrt(mean_squared_error(y_train, train_pred))    # RMSE

    metrics.append(score)
    print(f'\n{np.mean(metrics)}')
    return imp
