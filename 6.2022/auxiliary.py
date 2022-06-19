import os
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import wraps

from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error

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


def train_valid_split(df, *, frac, seed=None):
    """ Prepare data for validation
    :param df - original dataset
    :param frac - the percentage of data to be set in NaN
    :param seed - random seed to achieve repeatability
    :return True values and dataset with NaNs for validation
    """
    data = df.dropna().copy()
    exclude_fixed = df.isna().any()
    size = int(data.size * frac)
    np.random.seed(seed)

    nans = 0
    candidates = np.array([[],[]])
    while nans < size:
        columns = data.columns[~data.isna().all() & exclude_fixed]      # select columns containing notNA values & exclude 
        index = data.index[~data.isna().all(axis=1)]    # select rows containing notNA values
        # generate NaN positions
        col = np.random.choice(columns, size=size - nans, replace=True)
        idx = np.random.choice(index, size=size - nans, replace=True)

        candidates = np.hstack([candidates, np.vstack([idx, col])])
        nans = pd.DataFrame(candidates.T).drop_duplicates().shape[0]

    # drop chosen values
    chosen = pd.DataFrame(candidates.T).drop_duplicates()
    for col, idx, in chosen.groupby(1)[0].unique().iteritems():
        data.loc[np.sort(idx), col] = np.nan
    return df.dropna().copy(), data


def deprecated(msg):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Using `{func.__name__}()` is deprecated! {msg}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# =============================== Impute Helper Definition ===============================
class Step:
    def __init__(self, func, columns, **kwargs):
        """
        :param func - step function, must have signature like func(train, test, **params)
        :param columns - columns passed to function
        :keyword inherit - use data calculated in the previous step; default = True
        :keyword autosplit - automatically split data to train and test; default = True
        :keyword max_fill_nan_count - max allowed number of NaN in row to predict value in it; default = np.inf
        :keywird max_train_nan_count - max allowed number of NaN in row to use it in train. NaN will be filled in with mean; default = np.inf
        """

        assert func is not callable, "`func` parameter must be callable"
        self.func = func
        self.columns = columns
        # parse parameters
        self.max_fill_nan_count = kwargs.pop('max_fill_nan_count', np.inf)
        assert self.max_fill_nan_count >= 1, "`max_fill_nan_count` < 1 doesn't make sense. No value will be predicted."
        self.max_train_nan_count = kwargs.pop('max_train_nan_count', np.inf)
        assert self.max_train_nan_count >= 0, "`max_train_nan_count` < 0 doesn't make sense. No data left for training."
        self.inherit = kwargs.pop('inherit', True)
        self.autosplit = kwargs.pop('autosplit', True)
        self.params = kwargs

    def call(self, df):
        # train/test split
        if self.autosplit:
            self.columns = self.columns if self.columns != 'all' else df.columns
            fill_nan_count = df.isna().sum(axis=1) <= self.max_fill_nan_count
            train_nan_count = df.isna().sum(axis=1) <= self.max_train_nan_count        
            train = df.loc[train_nan_count, self.columns]
            test = df.loc[fill_nan_count, self.columns]
            return self.func(train, test, **self.params)
        else:
            return self.func(df, **self.params)


class ImputeHelper():
    def __init__(self, *steps):
        self.steps = steps

    def run(self, data, *, validate_on=None):
        """
        :params data - original dataset
        """
        predicted = data.copy()
        for step in self.steps:            
            # call step function
            step_result = step.call(predicted if step.inherit else data)
            predicted.fillna(step_result, inplace=True)
        if validate_on is not None:
            score, overall = calc_train_score(data, validate_on, predicted)
            print(f'Final validation score: {score}', f'Overall final score: {overall}', sep='\n')
        return predicted


def calc_train_score(df, true_df, pred):
    """
    :param df - dataset containing NaN
    :param true_df - dataset with true values
    :param pred - dataset with predicted values
    """
    # compute score
    metrics = []
    nans = df.isna()
    fully_predicted = ~pred.isna().any()
    # iterate through columns that originally contain NaN and are fully predicted
    for col in tqdm(df.columns[nans.any() & fully_predicted], desc='Final validation'):
        y_train = true_df.loc[nans[col], col]
        train_pred = pred.loc[nans[col], col]
        metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))
    # calc overall score
    columns = df.columns[~nans.any() | fully_predicted]
    overall = np.sqrt(mean_squared_error(true_df[columns], pred[columns]))
    return np.mean(metrics), overall


# =============================== Imputer step functions ===============================
def simplestat(train, test, imputer=SimpleImputer()):
    """ Impute data with SimpleImputer """
    imputer.fit(train)
    values = pd.DataFrame(imputer.transform(test), index=test.index, columns=test.columns)
    return test.fillna(values)


def groupstat(train, test, *, gcol, func='mean'):
    """ Impute data with grouped statistics """
    stats = train.groupby(gcol).transform(func)
    if stats.isna().any().any():
        print(f'Stats contain NaN! Data may not filled in completely!')
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


# ====== Mean of k nearest values (modification of Predictive Mean Matching) ======
# canonical PMM will be reached when using N=1
def mean_matching(train, test=None, *, N, init, backend=None):
    """ Mean of k nearest values (modification of Predictive Mean Matching)    
    To use it as canonical PMM set N=1.
    :param train - part of original data for calculating statistics or/and training initiator
    :prarm test - part of original data in which NaN will be filled in
    :param N - number of nearest values to use in statistic calculation
    :param init - NaN initiator. Estimator, transformer or fixed value which is used to fill NaN in the first approximation
    :param backend - parallel backend (for more information see sklearn docs)
    """
    if test is None:
        test = train
    test_nan = test.isna()

    if hasattr(init, 'predict'):         # initiator is an estimator
        initiated = test.copy()
        metrics = []
        for col in (pbar := tqdm(train.columns[train.isna().any()], desc='Initiate values')):
            # fit initiator
            X_train = train[~train[col].isna()].drop(col, axis=1)
            y_train = train.loc[~train[col].isna(), col]            
            init.fit(X_train, y_train)
            # calc initiator metric
            train_pred = init.predict(X_train)
            metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))
            # predict init values
            X_test = test[test[col].isna()].drop(col, axis=1)
            initiated.loc[test[col].isna(), col] = init.predict(X_test)
            pbar.set_postfix({'avg. score': np.mean(metrics)})
        # print(f'Initiator avg. score: {np.mean(metrics)}')
    elif isinstance(init, TransformerMixin):    # initiator is a transformer
        initiated = pd.DataFrame(init.fit(train).transform(test), index=test.index, columns=test.columns)
    else:                                       # initiator is a fixed value
        initiated = pd.DataFrame(test.fillna(init), index=test.index, columns=test.columns)
    # collect unique placeholders in each columns
    uniques = initiated.apply(lambda col: col[test_nan[col.name]].unique())

    def get_remapper(col_name):
        unique = uniques[col_name]
        if unique.size == 0:
            return {}
        column = train.loc[~train[col_name].isna(), col_name]
        col_idx = train.columns.get_loc(column.name)

        nearest = np.apply_along_axis(lambda val: np.abs(column - val).argsort()[:N], 0, [unique])      # collect N nearest items from train
        statistics = np.apply_along_axis(lambda idx: train.iloc[idx, col_idx].mean(), 0, nearest)       # calc statistics for these nearest items
        return dict(zip(unique, statistics))

    if backend is not None:
        with parallel_backend(backend):
            colmaps = (Parallel()(delayed(get_remapper)(col) for col in tqdm(test.columns, desc='Collect remapper')))
        remapper = dict(zip(test.columns, colmaps))
    else:
        remapper = {col: get_remapper(col) for col in tqdm(test.columns, desc='Collect remapper')}
    # fill in
    return initiated[test_nan].apply(lambda col: col.map(remapper[col.name], na_action='ignore')).fillna(initiated)
    # return initiated[nans].replace(remapper).fillna(initiated)


# ==================== Multiple imputation of chained equations ===================
def mice(df, test=None, *, estimator, epochs=10, seed=None):
    """ Multiple imputation by chained equations
    :param df - original dataset
    :param estimator - estimetor used for imputation
    :param epochs - number of iterations
    :param seed - random seed to achieve repeatability
    """
    assert test is None, "This MICE implementation requires the autosplit parameter to be set to False"
    
    # initiate
    data = df.fillna(df.mean())
    nans = df.isna()
    np.random.seed(seed)

    for n in range(epochs):
        epoch_metrics = []
        for col in (pbar := tqdm(data.columns[nans.any()], desc=f'Epoch {n + 1} / {epochs}')):
            # train/test split
            X_train = data[~nans[col]].drop(col, axis=1)
            y_train = data.loc[~nans[col], col]
            X_test = data[nans[col]].drop(col, axis=1)
            # fit/predict        
            estimator.set_params(random_state=np.random.randint(2**32))
            estimator.fit(X_train, y_train)
            train_pred = estimator.predict(X_train)
            epoch_metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))
            # update mising values
            data.loc[nans[col], col] = estimator.predict(X_test)
            pbar.set_postfix({'avg. score': np.mean(epoch_metrics)})
        # print(f'Epoch {n + 1} avg. score: {np.mean(epoch_metrics)}')
    return data


# =============================== Cosine similarity ===============================
# this takes a VERY long time
class CosineSimilarity:
    def __init__(self, train, test, *, seed=None):
        nan_rows = test.isna().any(axis=1)
        self.use_cols = train.columns[~train.isna().any() & ~test.isna().any()]
        self.train = train
        self.test = test[nan_rows]
        np.random.seed(seed)
    
    def _process_chunk(self, chunk, k, threshold, subsample):
        index = self.train.sample(frac=subsample).index
        cosine = cosine_similarity(chunk, self.train.loc[index, self.use_cols])
        if threshold is not None:           # collect by threshold
            mask = cosine > threshold       # apply threshold
            np.fill_diagonal(mask, False)   # exclude ownes
            # return list(map(lambda m: self.df[m].mean().values, mask))
            return np.apply_along_axis(lambda m: self.train.loc[index][m].mean(), 1, mask)
        elif k is not None:     # collect by k nearest
            # return list(map(lambda c: self.df.iloc[np.argsort(c)[::-1][1:]].head(k).mean().values, cosine))
            return np.apply_along_axis(lambda c: self.train.loc[index].iloc[np.argsort(c)[::-1][1:]].head(k).mean(), 1, cosine)

    def calculate(self, *, k=None, threshold=None, backend=None, chunksize=50, subsample=1.0):
        assert bool(k) ^ bool(threshold), 'Only one parameter must be specified: either `k` or `threshold`.'

        test_size = self.test.index.size
        chunk_count = test_size // chunksize + (1 if test_size % chunksize else 0)
        print(f'{test_size} rows in {chunk_count} chunks')
        arr = []
        if backend is not None:
            with parallel_backend(backend):
                result = (Parallel(n_jobs=-1)(
                    delayed(self._process_chunk)(self.test[self.use_cols].iloc[start:start + chunksize], k, threshold, subsample) 
                        for start in tqdm(range(0, test_size, chunksize), total=chunk_count)
                ))
        else:
            result = [self._process_chunk(self.test[self.use_cols].iloc[start:start + chunksize], k, threshold, subsample)
                        for start in tqdm(range(0, test_size, chunksize), total=chunk_count)]
        # parse result
        for part in result:
            arr.extend(part)
        stats = pd.DataFrame(arr, index=self.test.index, columns=self.test.columns)
        if stats.isna().any().any():
            print('Some values are NaN. Try decrease threshold.')
        return stats


@deprecated("It may take a VERY long time on large data, work unstable or don't work at all.")
def cosine_stats(train, test, *, k=None, threshold=None, backend=None, chunksize=50, subsample=1.0):
    if backend is not None:
        os.environ['MKL_NUM_THREADS'] = '1'
    csim = CosineSimilarity(train, test)
    stats = csim.calculate(k=k, threshold=threshold, backend=backend, chunksize=chunksize, subsample=subsample)
    pred = test.fillna(stats)
    return pred
