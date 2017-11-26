import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid

# file parameters
batch_size = 1024 * 128
train_files = ['20080101']
test_files = [
    '20080710', '20080725',
    '20080810', '20080825',
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

def preprocess(files, out_dir, features=None, shuffle=False):
    print('Processing {}'.format(out_dir))
    n_by_class = pd.Series().astype('int')
    n_batches_written = 0
    df_in_write = pd.DataFrame()

    n_files = len(files)
    for i, fname in enumerate(files):
        df = pd.read_csv(
            data_dir + 'raw/' + fname + '.txt',
            delim_whitespace=True, header=None,
            low_memory=False
        )

        df.drop(columns['drop'], axis=1, inplace=True)
        df.drop_duplicates(inplace=True)

        # for col in columns['count_of_100']:
        #     df[col] /= 100

        for col in columns['numerical']:
            if col not in num_enc:
                num_enc[col] = (df[col].mean(), df[col].std())
            mu, std = num_enc[col]
            df[col] = (df[col] - mu) / std
            df[col] = sigmoid(df[col].values)

        df = pd.get_dummies(df, columns=columns['categorical'])

        # df['label'] = [
        #     int(y < 0) for y in df[columns['label']].values
        # ]
        df['label'] = [
            label_enc.index(y) for y in df[columns['label']].values
        ]
        df.drop(columns['label'], axis=1, inplace=True)
        n_by_class = n_by_class.add(df['label'].value_counts(), fill_value=0)

        if features is None:
            features = df.columns
        else:
            df = df.reindex(columns = features, fill_value=0)

        df_in_write = df_in_write.append(df, ignore_index=True)
        if shuffle:
            df_in_write = df_in_write.sample(frac=1)

        endchar = '\n' if i == n_files-1 else '\r'
        print('Writing {} [{:2}/{:2}]'.format(fname, i+1, n_files), end=endchar)

        while len(df_in_write) >= batch_size:
            df_in_write.iloc[:batch_size].to_csv(
                data_dir + out_dir + str(n_batches_written) + '.csv',
                header=False, index=False
            )
            df_in_write.drop(
                df_in_write.index[:batch_size],
                axis=0, inplace=True
            )
            n_batches_written += 1

    remainder = len(df_in_write)
    if remainder > 0:
        df_in_write.to_csv(
            data_dir + out_dir + str(n_batches_written) + '.csv',
            header=False, index=False
        )
    n_rows_written = n_batches_written * batch_size + remainder

    info = pd.concat([
        pd.Series([batch_size, n_rows_written]),
        features.to_series()
    ])
    info.to_csv(
        data_dir + out_dir + 'info.csv',
        header=False, index=False
    )

    stats = n_by_class / n_by_class.sum()
    stats.to_csv(
        data_dir + out_dir + 'stats.csv',
        header=False, index=True
    )
    print('===> Stats for {}:'.format(files))
    print(stats)

    return features

train_features = preprocess(train_files, out_dir='train/')
_ = preprocess(test_files, out_dir='test/', features=train_features)

