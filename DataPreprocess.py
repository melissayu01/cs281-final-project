import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid

# file parameters
batch_size = 100000
train_files = ['20080101']
test_files = [
    '20080110', '20080125',
#     '20080710', '20080725',
#     '20080810', '20080825',
#     '20080910', '20080923',
#     '20081010', '20081025',
#     '20081110', '20081125',
#     '20081210', '20081225',
]
data_dir = '../data/'

# feature parameters
columns = {
    'drop': np.array([1, 2, 3, 5, 6, 7, 8, 9]) + 14 - 1,
    'count_of_100': np.array([9, 10]) - 1,
    'numerical':  np.array([1] + list(range(3, 14))) - 1,
    'categorical': np.array([2, 14, 10+14]) - 1, # (column_idx, n_states)
    'label': 4+14-1
}
label_enc = [1, -1, -2] # safe, known attack, unknown attack
num_enc = dict()

def preprocess(files, out_dir, features=None):
    n_batches_written = 0
    df_in_write = pd.DataFrame()
    for fname in files:
        print('processing {} to {}'.format(fname, out_dir))
        df = pd.read_csv(
            data_dir + 'raw/' + fname + '.txt',
            delim_whitespace=True, header=None
        )

        df.drop(columns['drop'], axis=1, inplace=True)

        for col in columns['count_of_100']:
            df[col] /= 100

        for col in columns['numerical']:
            if col not in num_enc:
                num_enc[col] = (df[col].mean(), df[col].std())
            mu, std = num_enc[col]
            df[col] = (df[col] - mu) / std
            df[col] = sigmoid(df[col].values)

        df = pd.get_dummies(df, columns=columns['categorical'])

        df['label'] = [label_enc.index(y) for y in df[columns['label']].values]
        df.drop(columns['label'], axis=1, inplace=True)

        if features is None:
            features = df.columns
        else:
            df = df.reindex(columns = features, fill_value=0)

        df_in_write = df_in_write.append(df, ignore_index=True)
        df_in_write = df_in_write.sample(frac=1)

        while len(df_in_write) >= batch_size:
            df_in_write.iloc[:batch_size].to_csv(
                data_dir + out_dir + str(n_batches_written) + '.csv',
                header=False, index=False
            )
            df_in_write.drop(df_in_write.index[:batch_size], axis=0, inplace=True)
            n_batches_written += 1

    remainder = len(df_in_write)
    if remainder > 0:
        df_in_write.to_csv(
            data_dir + out_dir + str(n_batches_written) + '.csv',
            header=False, index=False
        )
    n_rows_written = n_batches_written * batch_size + remainder

    info = pd.concat([pd.Series([batch_size, n_rows_written]), features.to_series()])
    info.to_csv(
        data_dir + out_dir + 'info.csv',
        header=False, index=False
    )

    return features

train_features = preprocess(train_files, out_dir='train/')
_ = preprocess(test_files, out_dir='test/', features=train_features)

