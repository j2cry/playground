import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import wraps

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
    print(f'Data contain NaN: {predicted.isna().any().any()}')
    print(f'Data contain inf: {((predicted == np.inf) | predicted == -np.inf).any().any()}')
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


# ================================= Decoration =================================
class parameter:
    class initiator:
        @classmethod
        def required(cls, mode='any'):
            def required_mode(func):
                @wraps(func)
                def wrapper(train, test, initiator, *args, **kwargs):
                    assert initiator is not None, f"This `{func.__name__}()` implementation requires an initiator."
                    if mode == 'predictor':
                        assert hasattr(initiator, 'predict'), f"This `{func.__name__}()` implementation requires an estimator as the initiator."
                    elif mode == 'transformer':
                        assert isinstance(initiator, TransformerMixin), f"This `{func.__name__}()` implementation requires a transformer as the initiator."
                    return func(train, test, initiator, *args, **kwargs)
                return wrapper
            return required_mode

        @classmethod
        def allowed(cls, mode='any'):
            def allowed_mode(func):
                @wraps(func)
                def wrapper(train, test, initiator, *args, **kwargs):
                    if initiator is not None:
                        if mode == 'predictor':
                            assert hasattr(initiator, 'predict'), f"This `{func.__name__}()` implementation allows only an estimator as the initiator."
                        elif mode == 'transformer':
                            assert isinstance(initiator, TransformerMixin), f"This `{func.__name__}()` implementation allows only a transformer as the initiator."
                        elif mode == 'any':
                            print("Using `@parameter.initiator.allowed('any')` is redundant")
                    return func(train, test, initiator, *args, **kwargs)
                return wrapper
            return allowed_mode


def deprecated(msg):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Using `{func.__name__}()` is deprecated! {msg}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ========================== Impute Helper Definition ==========================
class Step:
    def __init__(self, func, fit_columns, fill_columns=None, max_fit_nan_count=np.inf, max_fill_nan_count=np.inf, inherit=True, initiator=None, **kwargs):
        """ Imputer step description
        :param func - step function, must have signature like func(train, test, **params)
        :param fit_columns - columns to use in train
        :param fill_columns - columns to fill in. Must be a subset of `fit_columns`. Is the same as `fit_columns` if None
        :param max_fit_nan_count - max allowed number of NaN in the train row. Rows containing more NaN values are not passed to the function. (default = np.inf)
        :param max_fill_nan_count - max allowed number of NaN in the filling row. Rows containing more NaN values are not passed to the function. (default = np.inf)
        :param inherit - use data calculated in the previous step (default = True)
        :param initiator - strategy for initiating NaN
        """

        assert func is not callable, "`func` parameter must be callable"
        self.func = func
        self.fit_columns = fit_columns
        self.fill_columns = fill_columns if fill_columns is not None else fit_columns
        
        # parse parameters
        assert max_fit_nan_count >= 0, "`max_fit_nan_count` < 0 doesn't make sense. No data left for training."
        self.max_fit_nan_count = max_fit_nan_count
        assert max_fill_nan_count >= 1, "`max_fill_nan_count` < 1 doesn't make sense. No value will be predicted."
        self.max_fill_nan_count = max_fill_nan_count
        self.inherit = inherit
        self.initiator = initiator
        self.kwargs = kwargs

    def call(self, df):
        self.fit_columns = df.columns if self.fit_columns == 'all' else self.fit_columns
        self.fill_columns = df.columns if self.fill_columns == 'all' else self.fill_columns
        assert set(self.fill_columns).issubset(self.fit_columns), f"`{self.func.__name__}()`: fill_columns must be a subset of fit_columns."

        # check if fill columns contain NaN
        nan_fill = df[self.fill_columns].isna()
        assert nan_fill.any().any(), f"`{self.func.__name__}()`: No NaN values to fill in"

        # apply nan count filters
        fit_nan_count = df[self.fit_columns].isna().sum(axis=1) <= self.max_fit_nan_count
        fill_nan_count = nan_fill.sum(axis=1) <= self.max_fill_nan_count
        train = df.loc[fit_nan_count, self.fit_columns]     # select required train part
        test = df.loc[fill_nan_count, self.fit_columns]     # select required test part

        return self.func(train, test, initiator=self.initiator, fill_columns=self.fill_columns, **self.kwargs)


class ImputeHelper():
    def __init__(self, *steps):
        self.steps = steps

    def run(self, data, *, validate_on=None):
        """ Run imputer
        :params data - original dataset
        :validate_on - dataset for validation
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


def data_initialize(initiator, train, test, *, separately=True):
    """ Initialize missing values with given initiator """
    if separately:
        train_data = train
        test_data = test
    else:
        train_data = pd.concat([train, test], axis=0).drop_duplicates()
        test_data = train_data

    # initialize missing values    
    if hasattr(initiator, 'predict'):
        # initiator is an estimator
        train_init = train_data.copy()
        test_init = test_data.copy()
        metrics = []
        for col in (pbar := tqdm(train_data.columns[train_data.isna().any()], desc='Initiate values')):
            # fit initiator
            train_nan_rows = train_data[col].isna()
            X_train = train_data[~train_nan_rows].drop(col, axis=1)
            y_train = train_data.loc[~train_nan_rows, col]
            initiator.fit(X_train, y_train)
            # calc initiator metric
            train_pred = initiator.predict(X_train)
            metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))
            nan_X_train = train_init[train_nan_rows].drop(col, axis=1)
            train_init.loc[train_nan_rows, col] = initiator.predict(nan_X_train)

            # predict init values
            test_nan_rows = test_data[col].isna()
            X_test = test_data[test_nan_rows].drop(col, axis=1)
            test_init.loc[test_nan_rows, col] = initiator.predict(X_test)
            pbar.set_postfix({'avg. score': np.mean(metrics)})
    elif isinstance(initiator, TransformerMixin):
        # initiator is a transformer
        train_init = pd.DataFrame(initiator.fit_transform(train_data), index=train_data.index, columns=train_data.columns)
        test_init = pd.DataFrame(initiator.transform(test_data), index=test_data.index, columns=test_data.columns)
    else:
        # initiator is a fixed value
        train_init = pd.DataFrame(train_data.fillna(initiator), index=train_data.index, columns=train_data.columns)
        test_init = pd.DataFrame(test_data.fillna(initiator), index=test_data.index, columns=test_data.columns)
    # return initiated data and optionally concatenated data
    return (train_init, test_init) if separately else (train_init, train_data)

# =========================== Imputer step functions ===========================
@parameter.initiator.required('transformer')
def transformer(train, test, initiator, fill_columns, **kwargs):
    """ Impute data with given transformer """
    initiator.fit(train)
    values = pd.DataFrame(initiator.transform(test), index=test.index, columns=test.columns)
    return test[fill_columns].fillna(values)


@parameter.initiator.allowed('transformer')
def groupstat(train, test, initiator, fill_columns, *, gcol, func='mean'):
    """ Impute data with grouped statistics """
    train_init = train if initiator is None else pd.DataFrame(initiator.fit_transform(train), index=train.index, columns=train.columns)
    # TODO check fill_columns and gcol intersection
    assert len(set(gcol).intersection(fill_columns)) == 0, f"`gcol` cannot intersect with fill_columns."
    stats = train_init.groupby(gcol).transform(func)
    pred = test[fill_columns].fillna(stats)
    if pred.isna().any().any():
        print(f'Stats contain NaN, so data was not filled in completely!')
    return pred


@parameter.initiator.allowed('any')
def predictor(train, test, initiator, fill_columns, *, estimator, neural=False, **fit_params):
    """ Impute data with one-per-column estimators """
    # initialize missing values
    if initiator is None:
        train_init = train
        test_init = test
    else:
        train_init, test_init = data_initialize(initiator, train, test)

    pred = test.copy()
    metrics = []

    nan_cols = test.columns[test.isna().any()].intersection(fill_columns)      # select columns to be filled in
    for col in (pbar := tqdm(nan_cols)):
        nan_target = train[col].isna()      # exclude rows without target from train
        target_mask = test[col].isna()      # select rows that are NaN in this columns
        # train/test split
        X_train = train_init[~nan_target].drop(col, axis=1)
        y_train = train_init.loc[~nan_target, col]
        X_test = test_init[target_mask].drop(col, axis=1)
        # fit/predict
        estimator.fit(X_train, y_train, **fit_params)
        if neural:
            train_pred = estimator.predict(X_train).T[0]
            pred.loc[target_mask, col] = estimator.predict(X_test).T[0]
        else:
            train_pred = estimator.predict(X_train)
            pred.loc[target_mask, col] = estimator.predict(X_test)
        # score pipeline
        metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))    # RMSE
        pbar.set_postfix({'avg. score': np.mean(metrics)})
    return pred


@parameter.initiator.allowed('any')
def onestep_neural(train, test, initiator, fill_columns, *, model, **fit_params):
    """ Impute data with neural network with multidimensional output
    To one-per-column approach use predictor(..., neural=True)
    """
    # initialize missing values
    if initiator is None:
        train_init = train
        test_init = test
    else:
        train_init, test_init = data_initialize(initiator, train, test)

    pred = test.copy()
    nans = train[fill_columns].isna().any(axis=1)
    X_train = train_init[~nans].drop(fill_columns, axis=1)
    y_train = train_init.loc[~nans, fill_columns]
    target_mask = test[fill_columns].isna().any(axis=1)
    X_test = test_init[target_mask].drop(fill_columns, axis=1)

    # fit/predict
    model.fit(X_train, y_train, **fit_params)
    # train_pred = model.predict(X_train)
    # score = np.sqrt(mean_squared_error(y_train, train_pred))
    pred.loc[target_mask, fill_columns] = model.predict(X_test)
    return pred


# ===== Mean of k nearest values (modification of Predictive Mean Matching) ====
# canonical PMM will be reached when using N=1
@parameter.initiator.required('any')
def mean_matching(train, test, initiator, fill_columns, *, N, backend=None):
    """ Mean of k nearest values (modification of Predictive Mean Matching)    
    To use it as canonical PMM set N=1.
    :param train - part of original data for calculating statistics or/and training initiator
    :prarm test - part of original data in which NaN will be filled in
    :param initiator - strategy for initiating NaN
    :param N - number of nearest values to use in statistic calculation
    :param backend - parallel backend (for more information see sklearn docs)
    """
    _, test_init = data_initialize(initiator, train, test)

    # collect unique values in each fill_column
    fill_nan = test[fill_columns].isna()
    uniques = test_init[fill_columns].apply(lambda col: col[fill_nan[col.name]].unique())

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
            colmaps = (Parallel()(delayed(get_remapper)(col) for col in tqdm(fill_columns, desc='Collect remapper')))
        remapper = dict(zip(fill_columns, colmaps))
    else:
        remapper = {col: get_remapper(col) for col in tqdm(fill_columns, desc='Collect remapper')}
    # fill in
    return test_init[fill_columns].apply(lambda col: col.map(remapper[col.name], na_action='ignore')).fillna(test_init[fill_columns])
    # return initiated[nans].replace(remapper).fillna(initiated)


# ================== Multiple imputation of chained equations ==================
@parameter.initiator.required('any')
def mice(train, test, initiator, fill_columns, *, estimator, epochs=5, seed=None):
    """ Multiple imputation by chained equations
    :param train - part of original data for calculating statistics or/and training initiator
    :prarm test - part of original data in which NaN will be filled in
    :param initiator - strategy for initiating NaN
    :param estimator - estimetor used for imputation
    :param epochs - number of iterations
    :param seed - random seed to achieve repeatability
    """
    initiated, concatenated = data_initialize(initiator, train, test, separately=False)

    nans = concatenated[fill_columns].isna()
    np.random.seed(seed)
    # multiple imputation
    for n in range(epochs):
        epoch_metrics = []
        for col in (pbar := tqdm(fill_columns, desc=f'Epoch {n + 1} / {epochs}')):
            # train/test split
            X_train = initiated[~nans[col]].drop(col, axis=1)
            y_train = initiated.loc[~nans[col], col]
            X_test = initiated[nans[col]].drop(col, axis=1)
            # fit/predict        
            estimator.set_params(random_state=np.random.randint(2**32))
            estimator.fit(X_train, y_train)
            train_pred = estimator.predict(X_train)
            epoch_metrics.append(np.sqrt(mean_squared_error(y_train, train_pred)))
            # update mising values
            initiated.loc[nans[col], col] = estimator.predict(X_test)
            pbar.set_postfix({'avg. score': np.mean(epoch_metrics)})
    return initiated[fill_columns]


# ============================== Cosine similarity =============================
# this takes a VERY long time
class CosineSimilarity:
    def __init__(self, train, test, *, seed=None):
        self.use_cols = train.columns
        self.train = train
        self.test = test
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
@parameter.initiator.required('transformer')
def cosine_stats(train, test, initiator, fill_columns, *, k=None, threshold=None, backend=None, chunksize=50, subsample=1.0, seed=None):
    # initialize missing values
    if initiator is None:
        train_init = train
        test_init = test
    else:
        train_init = pd.DataFrame(initiator.fit_transform(train), index=train.index, columns=train.columns)
        test_init = pd.DataFrame(initiator.transform(test), index=test.index, columns=test.columns)

    # if backend is not None:
        # os.environ['MKL_NUM_THREADS'] = '1'
    csim = CosineSimilarity(train_init, test_init, seed=seed)
    stats = csim.calculate(k=k, threshold=threshold, backend=backend, chunksize=chunksize, subsample=subsample)
    return test[fill_columns].fillna(stats)
