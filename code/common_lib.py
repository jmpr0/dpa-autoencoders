import os
from time import process_time as time

import numpy as np
import pandas as pd
import scipy
import sklearn
import tensorflow as tf
from callbacks import TimeEpochs
from scipy.sparse.linalg import expm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

str_to_int = dict([
    ("Active Wiretap", 0), ("ARP MitM", 1), ("Fuzzing", 2), ("OS Scan", 3), ("SSDP Flood", 4),
    ("SSL Renegotiation", 5), ("SYN DoS", 6), ("Video Injection", 7), ("Mirai", 8)
])

int_to_str = dict([(str_to_int[k], k) for k in str_to_int])


def load_dataset_n_basedir(basedir, attack=None, use_dup=True):
    train_data_n = pd.read_csv(
        '%s/Statistiche/Biflussi_Totali/training_set_%s.csv' % (basedir, 'all' if use_dup else 'nodup')
    )
    train_data_n = train_data_n[train_data_n['num_pkt'] > 1]

    if attack is not None:
        if isinstance(attack, (str, int)):
            if isinstance(attack, str):
                attack = str_to_int.get(attack, None)
                assert attack is not None, 'Error: available attacks are %s' % list(str_to_int.keys())
            train_data_n = train_data_n[train_data_n['Class'] == attack]
            basedir = '%s/%s' % (basedir, int_to_str[attack])
        elif isinstance(attack, list):
            for i in range(len(attack)):
                if isinstance(attack[i], str):
                    attack[i] = str_to_int.get(attack[i], None)
                    assert attack[i] is not None, 'Error: available attacks are %s' % list(str_to_int.keys())
            train_data_n = train_data_n[train_data_n['Class'].isin(attack)]
            basedir = '%s/%s' % (basedir, '_'.join([int_to_str[a] for a in attack]))
    else:
        basedir = '%s/All' % basedir
    return train_data_n, basedir


def dataset_split(n_samples, y=None, k=10, random_state=0):
    """
    :param n_samples: number of samples compose the dataset
    :param y: labels, for stratified splitting
    :param k: number of splits
    :param random_state: random state for the splitting function
    :return:
    """
    if y is None:
        splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
    else:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    indexes = [(train_index, test_index) for train_index, test_index in splitter.split(np.zeros((n_samples, 1)), y)]
    return indexes


def n_features_selection(x, y=None, method='amgm', feature_mode='max'):
    num_feats = [x.shape[1]] + list(range(5, x.shape[1], 5))[::-1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_norm = scaler.fit_transform(x)

    if method == 'amgm':
        def amgm(x):
            """
            paper: Efficient feature selection filters for high-dimensional data
            """
            return 1 / len(x) * 1 / np.exp(np.mean(x)) * np.sum(np.exp(x))

        scores = [amgm(feat) for feat in X_norm.T]
        sorted_indexes = list(np.argsort(scores))
        if feature_mode == 'max':
            sorted_indexes = sorted_indexes[::-1]

    elif method == 'lse':
        def construct_W(X, neighbour_size=5, t=1):
            S = kneighbors_graph(X, neighbour_size + 1, mode='distance',
                                 metric='euclidean')  # sqecludian distance works only with mode=connectivity  results were absurd
            S = (-1 * (S * S)) / (2 * t * t)
            S = S.tocsc()
            S = expm(S)  # exponential
            S = S.tocsr()
            # [1]  M. Belkin and P. Niyogi, “Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering,” Advances in Neural Information Processing Systems,
            # Vol. 14, 2001. Following the paper to make the weights matrix symmetrix we use this method
            bigger = np.transpose(S) > S
            S = S - S.multiply(bigger) + np.transpose(S).multiply(bigger)
            return S

        def LaplacianScore(X, neighbour_size=5, t=1):
            W = construct_W(X, t=t, neighbour_size=neighbour_size)
            n_samples, n_features = np.shape(X)

            # construct the diagonal matrix
            D = np.array(W.sum(axis=1))
            D = scipy.sparse.diags(np.transpose(D), [0])
            # construct graph Laplacian L
            L = D - W.toarray()

            # construct 1= [1,···,1]'
            I = np.ones((n_samples, n_features))

            # construct fr' => fr= [fr1,...,frn]'
            Xt = np.transpose(X)

            # construct fr^=fr-(frt D I/It D I)I
            t = np.matmul(np.matmul(Xt, D.toarray()), I) / np.matmul(np.matmul(np.transpose(I), D.toarray()), I)
            t = t[:, 0]
            t = np.tile(t, (n_samples, 1))
            fr = X - t

            # Compute Laplacian Score
            fr_t = np.transpose(fr)
            Lr = np.matmul(np.matmul(fr_t, L), fr) / np.matmul(np.dot(fr_t, D.toarray()), fr)

            return np.diag(Lr)

        def distanceEntropy(d, mu=0.5, beta=10):
            """
            As per: An Unsupervised Feature Selection Algorithm: Laplacian Score Combined with
            Distance-based Entropy Measure, Rongye Liu
            """
            if d <= mu:
                result = (np.exp(beta * d) - np.exp(0)) / (np.exp(beta * mu) - np.exp(0))
            else:
                result = (np.exp(beta * (1 - d)) - np.exp(0)) / (np.exp(beta * (1 - mu)) - np.exp(0))
            return result

        def lse(data, ls=None):
            """
            This method takes as input a dataset, its laplacian scores for all features
            and applies distance based entropy feature selection in order to identify
            the best subset of features in the laplacian sense.
            """
            if ls is None:
                ls = LaplacianScore(data)

            orderedFeatures = np.argsort(ls)
            scores = {}
            for i in range(2, len(ls)):
                selectedFeatures = orderedFeatures[:i]
                selectedFeaturesDataset = data[:, selectedFeatures]
                d = sklearn.metrics.pairwise_distances(selectedFeaturesDataset, metric='euclidean')
                beta = 10
                mu = 0.5

                d = MinMaxScaler().fit_transform(d)
                e = np.vectorize(distanceEntropy)(d, mu, beta)
                e = MinMaxScaler().fit_transform(e)
                totalEntropy = np.sum(e)
                scores[i] = totalEntropy

            bestFeatures = orderedFeatures[:list(scores.keys())[np.argmin(scores.values())]]
            return bestFeatures

        for n in [10, 50, 100, 500, 1000, 5000, 10000, 50000]:
            t = time()
            best_features = lse(X_norm[:n])
            print(n, '\t', time() - t)

        print(best_features)
        exit()

    else:
        raise NotImplementedError('Method %s is not supported.' % method)

    if method in ['amgm']:  # Raking-based methods
        selected_features = []
        for n_feats in num_feats:
            selected_features.append(sorted_indexes[:n_feats])
    else:
        pass

    features = dict()
    for i in range(len(selected_features)):
        features[num_feats[i]] = {
            'selected_features': int
        }
        features[num_feats[i]] = selected_features[i]

    return features


def get_callbacks(outdir, discr='', early_stopping=False):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=0 if early_stopping else int(1e4)),
        TimeEpochs(),
        tf.keras.callbacks.ModelCheckpoint('%s/model%s.hdf5' % (outdir, discr),
                                           monitor='accuracy',
                                           mode='max', save_weights_only=True),
        tf.keras.callbacks.CSVLogger('%s/log%s.csv' % (outdir, discr))
    ]


def to_csv_bk(df, out_fn, **kwargs):
    if os.path.exists(out_fn):
        tmp_fn = '%s.bk' % out_fn
        count = 0
        while os.path.exists('%s%s' % (tmp_fn, count)):
            count += 1
        while count > 0:
            os.rename('%s%s' % (tmp_fn, count - 1), '%s%s' % (tmp_fn, count))
            count -= 1
        os.rename(out_fn, '%s%s' % (tmp_fn, count))
    df.to_csv(out_fn, **kwargs)
