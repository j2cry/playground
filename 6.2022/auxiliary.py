import pathlib
import pandas as pd

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
