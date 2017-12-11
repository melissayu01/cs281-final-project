import os, shutil
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid

# file parameters
BATCH_SZ = 1024 * 128
train_files = ['20080101']
test_files = [
    '20080710', '20080725',
    '20080810', '20080825',
    '20080910', '20080923',
    '20081010', '20081025',
    '20081110', '20081125',
    '20081210', '20081225',
]
data_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, 'data'))

# feature parameters
columns = {
    'drop': np.array([5, 6, 7, 8, 9]) + 14 - 1,
    'detectors': np.array([1, 2, 3]) + 14 - 1,
    'count_of_100': np.array([9, 10]) - 1,
    'numerical':  np.array([1] + list(range(3, 14))) - 1,
    'categorical': np.array([2, 14, 10+14]) - 1,
    'label': 4+14-1
}
label_enc = [1, -1, -2] # safe, known attack, unknown attack
num_enc = dict()

def clear_folder(folder):
    for f in os.listdir(folder):
        file_path = os.path.join(folder, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def preprocess(files, out_dir,
               features=None, shuffle=False,
               remove_dups=False, binary_labels=False):
    print(
        'Processing {} with [shuffle {}] [remove_dups {}] [binary_labels {}]'
        .format(out_dir, shuffle, remove_dups, binary_labels))

    n_files = len(files)
    n_by_class = pd.Series().astype('int')
    n_batches_written = 0
    df_in_write = pd.DataFrame()
    parse_detection_codes = lambda code: int(code != '0')

    # initialize data folder
    write_path = os.path.join(data_dir, out_dir)
    clear_folder(write_path)

    for i, fname in enumerate(files):
        df = pd.read_csv(
            os.path.join(data_dir, 'raw', (fname + '.txt')),
            delim_whitespace=True, header=None, low_memory=False,
            converters={
                i: parse_detection_codes
                for i in columns['detectors']
            }
        )

        df.drop(columns['drop'], axis=1, inplace=True)
        if remove_dups:
            df.drop_duplicates(inplace=True)

        for col in columns['numerical']:
            if col not in num_enc:
                num_enc[col] = (df[col].mean(), df[col].std())
            mu, std = num_enc[col]
            df[col] = (df[col] - mu) / std
            df[col] = sigmoid(df[col].values)

        df = pd.get_dummies(df, columns=columns['categorical'])

        if binary_labels:
            df['label'] = [int(y < 0) for y in df[columns['label']].values]
        else:
            df['label'] = [
                label_enc.index(y) for y in df[columns['label']].values
            ]
        df.drop(columns['label'], axis=1, inplace=True)
        n_by_class = n_by_class.add(df['label'].value_counts(), fill_value=0)

        if features is None:
            features = df.columns
        else:
            df = df.reindex(columns = features, fill_value=0)

        if remove_dups:
            df.drop_duplicates(inplace=True)

        df_in_write = df_in_write.append(df, ignore_index=True)
        if shuffle:
            df_in_write = df_in_write.sample(frac=1)

        endchar = '\n' if i == n_files-1 else '\r'
        print('{} [{:2}/{:2}]'.format(fname, i+1, n_files), end=endchar)

        while len(df_in_write) >= BATCH_SZ:
            df_in_write.iloc[:BATCH_SZ].to_csv(
                os.path.join(write_path, '{}.csv'.format(n_batches_written)),
                header=False, index=False
            )
            df_in_write.drop(
                df_in_write.index[:BATCH_SZ],
                axis=0, inplace=True
            )
            n_batches_written += 1

    remainder = len(df_in_write)
    if remainder > 0:
        df_in_write.to_csv(
            os.path.join(write_path, '{}.csv'.format(n_batches_written)),
            header=False, index=False
        )
    n_rows_written = n_batches_written * BATCH_SZ + remainder

    info = pd.concat([
        pd.Series([BATCH_SZ, n_rows_written]),
        features.to_series()
    ])
    info.to_csv(
        os.path.join(write_path, 'info.csv'),
        header=False, index=False
    )

    stats = n_by_class / n_by_class.sum()
    stats.to_csv(
        os.path.join(write_path, 'stats.csv'),
        header=False, index=True
    )
    print('===> Stats for {}:'.format(files))
    print(stats)

    return features

train_features = preprocess(train_files, out_dir='train',
                            shuffle=True, remove_dups=True,
                            binary_labels=True)
_ = preprocess(test_files[::-1], out_dir='test',
               features=train_features, remove_dups=False,
               binary_labels=True)
