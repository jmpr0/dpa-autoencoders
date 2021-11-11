import math
import os
import sys
from ast import literal_eval
from time import process_time as time

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from common_lib import *
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential


def Kitsune_Features(basedir='.', outdir=None, epochs=[1], attack=None, use_dup=False, stratify=True,
                     only_metadata=False, nfeats=79, feature_mode='max', early_stopping=False):
    train_data_n, basedir = load_dataset_n_basedir(basedir, attack, use_dup)

    if outdir is not None:
        basedir = '%s/%s' % (outdir, basedir.replace('../', ''))

    feature_names = np.array(train_data_n.columns)[:nfeats]

    train_data_n = np.array(train_data_n)
    train_data_labels = train_data_n[:, -2]
    train_data_stratify = train_data_n[:, -1]
    train_data_stratify = np.array([s if l == 0 else s + 100 for l, s in zip(train_data_labels, train_data_stratify)])
    train_data_n = train_data_n[:, :nfeats]

    for iteration, (train_index, test_index) in enumerate(
            dataset_split(len(train_data_n), train_data_stratify if stratify else None, random_state=0)
    ):
        train_index = train_index.astype('int32')
        test_index = test_index.astype('int32')
        x_train = train_data_n[train_index]
        y_train = train_data_labels[train_index]
        x_test = train_data_n[test_index]
        y_test = train_data_labels[test_index]
        y_test = y_test.astype('int')

        # Filtering out malicious traffic
        benign_train_index = (y_train == 0)
        benign_test_index = (y_test == 0)
        x_train = x_train[benign_train_index]
        y_train = y_train[benign_train_index]

        selected_features = n_features_selection(x_train, y_train, feature_mode=feature_mode)

        # Training and Test against number of selected features
        for n in selected_features:
            features = selected_features[n]

            # Building of train and test sets given features
            training_features = x_train[:, features]
            test_features = x_test[:, features]

            n_autoencoder = 5
            n_autoencoder1 = n % n_autoencoder
            n_features2 = math.floor(n / n_autoencoder)
            n_features1 = n_features2 + 1
            n_autoencoder2 = n_autoencoder - n_autoencoder1

            for epoch in epochs:

                outdir = "%s/Training/Kitsune/fix/Epochs%s/Fold%s" % (basedir, epoch, iteration + 1)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                # Storing metadata
                df_meta_tr = pd.DataFrame(columns=['Malign', 'Class'], index=range(len(x_train)))
                df_meta_te = pd.DataFrame(columns=['Malign', 'Class'], index=range(len(x_test)))
                df_sel_feature_names = pd.DataFrame(columns=feature_names[features])
                df_meta_tr['Malign'] = y_train
                df_meta_tr['Class'] = [v if v < 100 else v - 100 for v in
                                       train_data_stratify[train_index][benign_train_index]]
                df_meta_te['Malign'] = y_test
                df_meta_te['Class'] = [v if v < 100 else v - 100 for v in train_data_stratify[test_index]]

                # Model Creation and Training
                Ensemble1 = np.empty(n_autoencoder1, dtype=object)  # Ensemble layer1
                Ensemble2 = np.empty(n_autoencoder2, dtype=object)  # Ensemble layer2
                for i in range(n_autoencoder1):
                    Ensemble1[i] = Sequential()
                    Ensemble1[i].add(
                        Dense(units=round(0.75 * n_features1), activation='relu', input_shape=(n_features1,)))
                    Ensemble1[i].add(Dense(units=n_features1, activation='sigmoid'))
                    Ensemble1[i].compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
                for i in range(n_autoencoder2):
                    Ensemble2[i] = Sequential()
                    Ensemble2[i].add(
                        Dense(units=round(0.75 * n_features2), activation='relu', input_shape=(n_features2,)))
                    Ensemble2[i].add(Dense(units=n_features2, activation='sigmoid'))
                    Ensemble2[i].compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
                # Creo un unico autoencoder di output che ha una dimensione di ingresso pari al n totale di autoencoder nell'ensemble layer
                Output = Sequential()
                Output.add(Dense(units=round(0.75 * n_autoencoder), activation='relu', input_shape=(n_autoencoder,)))
                Output.add(Dense(units=n_autoencoder, activation='sigmoid'))
                Output.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

                # Procedure to evaluate model inference time
                df_params_inference_info = pd.DataFrame(
                    columns=['Model Name', 'Trainable Parameters', 'Inference Time [s]']
                )
                n_dummy = 500
                X = np.random.random((n_dummy, n_features1))
                for i, model in enumerate(Ensemble1):
                    ts = []
                    for _ in range(10):
                        t = time()
                        model.predict(X)
                        t = time() - t
                        ts.append(t / n_dummy)
                    t = np.median(ts)
                    df_params_inference_info = df_params_inference_info.append(
                        {
                            'Model Name': 'Ensemble%s' % i,
                            'Trainable Parameters': sum([np.prod(v.shape) for v in model.trainable_variables]),
                            'Inference Time [s]': t
                        }, ignore_index=True
                    )
                X = np.random.random((n_dummy, n_features2))
                for i, model in enumerate(Ensemble2):
                    ts = []
                    for _ in range(10):
                        t = time()
                        model.predict(X)
                        t = time() - t
                        ts.append(t / n_dummy)
                    t = np.median(ts)
                    df_params_inference_info = df_params_inference_info.append(
                        {
                            'Model Name': 'Ensemble%s' % (n_autoencoder1 + i),
                            'Trainable Parameters': sum([np.prod(v.shape) for v in model.trainable_variables]),
                            'Inference Time [s]': t
                        }, ignore_index=True
                    )
                X = np.random.random((n_dummy, n_autoencoder))
                ts = []
                for _ in range(10):
                    t = time()
                    Output.predict(X)
                    t = time() - t
                    ts.append(t / n_dummy)
                t = np.median(ts)
                df_params_inference_info = df_params_inference_info.append(
                    {
                        'Model Name': 'Output',
                        'Trainable Parameters': sum([np.prod(v.shape) for v in Output.trainable_variables]),
                        'Inference Time [s]': t
                    }, ignore_index=True
                )

                df_meta_tr.to_csv('%s/training_labels_%s.csv' % (outdir, n), index=False)
                df_meta_te.to_csv('%s/test_labels_%s.csv' % (outdir, n), index=False)
                df_sel_feature_names.to_csv('%s/selected_features_%s.csv' % (outdir, n), index=False)
                df_params_inference_info.to_csv('%s/params_n_inference_%s.csv' % (outdir, n), index=False)

                if only_metadata:
                    continue

                # Dataset normalization
                scaler1 = MinMaxScaler(feature_range=(0, 1))
                training_features = scaler1.fit_transform(training_features)
                test_features = scaler1.transform(test_features)
                # Ensemble AE fitting
                for i in range(n_autoencoder1):
                    features_index = np.r_[i * n_features1:(i + 1) * n_features1]
                    Ensemble1[i].fit(
                        training_features[:, features_index], training_features[:, features_index],
                        epochs=epoch, batch_size=32,
                        callbacks=get_callbacks(outdir, '_%s_ensemble%s' % (n, i), early_stopping),
                        validation_data=(
                            test_features[benign_test_index][:, features_index],
                            test_features[benign_test_index][:, features_index]
                        )
                    )
                for i in range(n_autoencoder2):
                    features_index = np.r_[
                                     n_autoencoder1 * n_features1 + i * n_features2:n_autoencoder1 * n_features1 + (
                                             i + 1) * n_features2
                                     ]
                    Ensemble2[i].fit(
                        training_features[:, features_index], training_features[:, features_index],
                        epochs=epoch, batch_size=32,
                        callbacks=get_callbacks(outdir, '_%s_ensemble%s' % (n, n_autoencoder1 + i), early_stopping),
                        validation_data=(
                            test_features[benign_test_index][:, features_index],
                            test_features[benign_test_index][:, features_index]
                        )
                    )
                score = np.zeros((training_features.shape[0], n_autoencoder))
                test_score = np.zeros((test_features.shape[0], n_autoencoder))
                # Ensemble AEs RMSE computing
                for j in range(n_autoencoder1):
                    features_index = np.r_[j * n_features1:(j + 1) * n_features1]
                    pred = Ensemble1[j].predict(training_features[:, features_index])
                    test_pred = Ensemble1[j].predict(test_features[:, features_index])
                    score[:, j] = metrics.mean_squared_error(
                        pred.T, training_features[:, features_index].T, squared=False, multioutput='raw_values'
                    )
                    test_score[:, j] = metrics.mean_squared_error(
                        test_pred.T, test_features[:, features_index].T, squared=False,
                        multioutput='raw_values'
                    )
                for j in range(n_autoencoder2):
                    features_index = np.r_[
                                     n_autoencoder1 * n_features1 + j * n_features2:n_autoencoder1 * n_features1 + (
                                             j + 1) * n_features2
                                     ]
                    pred = Ensemble2[j].predict(training_features[:, features_index])
                    test_pred = Ensemble2[j].predict(test_features[:, features_index])
                    score[:, n_autoencoder1 + j] = metrics.mean_squared_error(
                        pred.T, training_features[:, features_index].T, squared=False, multioutput='raw_values'
                    )
                    test_score[:, n_autoencoder1 + j] = metrics.mean_squared_error(
                        test_pred.T, test_features[:, features_index].T, squared=False,
                        multioutput='raw_values'
                    )
                # Output AE training over Ensemble RMSEs
                scaler2 = MinMaxScaler(feature_range=(0, 1))
                score = scaler2.fit_transform(score)
                test_score = scaler2.transform(test_score)

                Output.fit(
                    score, score, epochs=epoch, batch_size=32,
                    callbacks=get_callbacks(outdir, '_%s_output' % n, early_stopping),
                    validation_data=(test_score[benign_test_index], test_score[benign_test_index])
                )

                # RMSE computing
                RMSE = np.zeros(test_score.shape[0])
                pred = Output.predict(test_score)
                for i in range(test_score.shape[0]):
                    RMSE[i] = np.sqrt(metrics.mean_squared_error(pred[i], test_score[i]))

                fpr, tpr, thresholds = metrics.roc_curve(y_test, RMSE)
                indices = np.where(fpr >= 0.01)
                index = np.min(indices)
                soglia = thresholds[index]
                labels = np.zeros(RMSE.shape[0])
                for i in range(RMSE.shape[0]):
                    if RMSE[i] < soglia:
                        labels[i] = 0
                    else:
                        labels[i] = 1
                for_plot = {'y_true': y_test, 'y_pred': labels, 'RMSE': RMSE}
                plot_df = pd.DataFrame.from_dict(for_plot)

                plot_df.to_csv(outdir + "/results_%s.csv" % n)


def Kitsune_Poisoning(basedir='.', outdir=None, attack=None, use_dup=False, stratify=True, only_metadata=False,
                      nfeats=79, early_stopping=False):
    train_data_n, basedir = load_dataset_n_basedir(basedir, attack, use_dup)

    if outdir is not None:
        basedir = '%s/%s' % (outdir, basedir.replace('../', ''))

    train_data_n = np.array(train_data_n)
    train_data_labels = train_data_n[:, -2]
    train_data_stratify = train_data_n[:, -1]
    train_data_stratify = np.array([s if l == 0 else s + 100 for l, s in zip(train_data_labels, train_data_stratify)])
    train_data_n = train_data_n[:, :nfeats]

    for iteration, (train_index, test_index) in enumerate(
            dataset_split(len(train_data_n), train_data_stratify if stratify else None, random_state=0)):
        malign_samples_per_config = None
        for percent_poisoning in (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10):
            train_index = train_index.astype('int32')
            test_index = test_index.astype('int32')
            x_train1 = train_data_n[train_index]
            y_train = train_data_labels[train_index]
            x_test = train_data_n[test_index]
            y_test = train_data_labels[test_index]
            y_test = y_test.astype('int')

            benign_index = (y_train == 0)
            malign_index = (y_train == 1)

            # Poisoning of the dataset
            benign_x_train = x_train1[benign_index]
            malign_x_train = x_train1[malign_index]
            malign_train_index = train_index[malign_index]

            y_train_stratify = train_data_stratify[train_index]
            malign_y_train_stratify = y_train_stratify[malign_index]

            n_malign_samples = int(np.ceil(benign_x_train.shape[0] * percent_poisoning / 100))

            # We incrementally select malicious samples
            if malign_samples_per_config is None:
                malign_x_train_resampled, malign_train_index_resampled = resample(
                    malign_x_train, malign_train_index, replace=False, n_samples=n_malign_samples,
                    random_state=0, stratify=malign_y_train_stratify if stratify else None
                )
            else:
                n_malign = n_malign_samples - len(malign_samples_per_config[0])
                malign_x_train_resampled, malign_train_index_resampled = resample(
                    malign_x_train[malign_samples_per_config[2]],
                    malign_train_index[malign_samples_per_config[2]],
                    replace=False, n_samples=n_malign, random_state=0,
                    stratify=malign_y_train_stratify[malign_samples_per_config[2]] if stratify else None)
                malign_x_train_resampled = np.concatenate(
                    (malign_samples_per_config[0], malign_x_train_resampled), axis=0
                )
                malign_train_index_resampled = np.concatenate(
                    (malign_samples_per_config[1], malign_train_index_resampled), axis=0
                )

            malign_x_train_resampled_l = malign_x_train_resampled.tolist()
            remaining_indexes = [
                i for i, v in enumerate(malign_x_train) if v.tolist() not in malign_x_train_resampled_l
            ]
            malign_samples_per_config = (malign_x_train_resampled, malign_train_index_resampled, remaining_indexes)

            x_train = np.concatenate([benign_x_train, malign_x_train_resampled], axis=0)
            np.random.shuffle(x_train)

            outdir = "%s/Training/Kitsune/Poisoning/%s%%/Fold%s" % (basedir, percent_poisoning, iteration + 1)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # Storing metadata
            df_meta_tr = pd.DataFrame(columns=['Malign', 'Class'], index=range(len(x_train)))
            df_meta_te = pd.DataFrame(columns=['Malign', 'Class'], index=range(len(x_test)))
            df_meta_tr['Malign'] = [0] * sum(benign_index) + [1] * n_malign_samples
            df_meta_tr['Class'] = np.concatenate(
                (
                    y_train_stratify[benign_index],
                    [v if v < 100 else v - 100 for v in train_data_stratify[malign_train_index_resampled]]
                ), axis=0
            )
            df_meta_te['Malign'] = y_test
            df_meta_te['Class'] = [v if v < 100 else v - 100 for v in train_data_stratify[test_index]]
            df_meta_tr.to_csv('%s/training_labels.csv' % outdir, index=False)
            df_meta_te.to_csv('%s/test_labels.csv' % outdir, index=False)

            if only_metadata:
                continue

            # Model Creation and Training
            training_features = x_train
            test_features = x_test
            n_autoencoder = 5
            n_autoencoder1 = x_train.shape[1] % n_autoencoder
            n_features2 = math.floor(x_train.shape[1] / n_autoencoder)
            n_features1 = n_features2 + 1
            n_autoencoder2 = n_autoencoder - n_autoencoder1
            Ensemble1 = np.empty(n_autoencoder1, dtype=object)  # Ensemble layer1
            Ensemble2 = np.empty(n_autoencoder2, dtype=object)  # Ensemble layer2
            for i in range(n_autoencoder1):
                Ensemble1[i] = Sequential()
                Ensemble1[i].add(Dense(units=round(0.75 * n_features1), activation='relu', input_shape=(n_features1,)))
                Ensemble1[i].add(Dense(units=n_features1, activation='sigmoid'))
                Ensemble1[i].compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            for i in range(n_autoencoder2):
                Ensemble2[i] = Sequential()
                Ensemble2[i].add(Dense(units=round(0.75 * n_features2), activation='relu', input_shape=(n_features2,)))
                Ensemble2[i].add(Dense(units=n_features2, activation='sigmoid'))
                Ensemble2[i].compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            Output = Sequential()
            Output.add(Dense(units=round(0.75 * n_autoencoder), activation='relu', input_shape=(n_autoencoder,)))
            Output.add(Dense(units=n_autoencoder, activation='sigmoid'))
            Output.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

            scaler1 = MinMaxScaler(feature_range=(0, 1))
            training_features = scaler1.fit_transform(training_features)
            for i in range(n_autoencoder1):
                Ensemble1[i].fit(training_features[:, i * n_features1:(i + 1) * n_features1],
                                 training_features[:, i * n_features1:(i + 1) * n_features1],
                                 epochs=100, batch_size=32,
                                 callbacks=get_callbacks(outdir, '_ensemble%s' % i, early_stopping))
            for i in range(n_autoencoder2):
                Ensemble2[i].fit(
                    training_features[:, n_autoencoder1 * n_features1 + i * n_features2:n_autoencoder1 * n_features1 + (
                            i + 1) * n_features2],
                    training_features[:, n_autoencoder1 * n_features1 + i * n_features2:n_autoencoder1 * n_features1 + (
                            i + 1) * n_features2], epochs=100, batch_size=32,
                    callbacks=get_callbacks(outdir, '_ensemble%s' % (n_autoencoder1 + i), early_stopping))
            score = np.zeros((training_features.shape[0], n_autoencoder))
            for j in range(n_autoencoder1):
                pred = Ensemble1[j].predict(training_features[:, j * n_features1:(j + 1) * n_features1])
                for i in range(training_features.shape[0]):
                    score[i, j] = np.sqrt(metrics.mean_squared_error(
                        pred[i], training_features[i, j * n_features1:(j + 1) * n_features1]))
            for j in range(n_autoencoder2):
                pred = Ensemble2[j].predict(
                    training_features[:, n_autoencoder1 * n_features1 + j * n_features2:n_autoencoder1 * n_features1 + (
                            j + 1) * n_features2])
                for i in range(training_features.shape[0]):
                    score[i, j + n_autoencoder1] = np.sqrt(
                        metrics.mean_squared_error(
                            pred[i], training_features[
                                     i, n_autoencoder1 * n_features1 + j * n_features2:n_autoencoder1 * n_features1 + (
                                    j + 1) * n_features2]))
            scaler2 = MinMaxScaler(feature_range=(0, 1))
            score = scaler2.fit_transform(score)
            Output.fit(score, score, epochs=100, batch_size=32,
                       callbacks=get_callbacks(outdir, '_output', early_stopping))
            RMSE = np.zeros(score.shape[0])
            pred = Output.predict(score)
            for i in range(score.shape[0]):
                RMSE[i] = np.sqrt(metrics.mean_squared_error(pred[i], score[i]))
            test_features = scaler1.transform(test_features)
            test_score = np.zeros((test_features.shape[0], n_autoencoder))
            for j in range(n_autoencoder1):
                pred = Ensemble1[j].predict(test_features[:, j * n_features1:(j + 1) * n_features1])
                for i in range(test_features.shape[0]):
                    test_score[i, j] = np.sqrt(
                        metrics.mean_squared_error(pred[i], test_features[i, j * n_features1:(j + 1) * n_features1]))
            for j in range(n_autoencoder2):
                pred = Ensemble2[j].predict(
                    test_features[:, n_autoencoder1 * n_features1 + j * n_features2:n_autoencoder1 * n_features1 + (
                            j + 1) * n_features2])
                for i in range(test_features.shape[0]):
                    test_score[i, j + n_autoencoder1] = np.sqrt(
                        metrics.mean_squared_error(
                            pred[i],
                            test_features[i,
                            n_autoencoder1 * n_features1 + j * n_features2:n_autoencoder1 * n_features1 + (
                                    j + 1) * n_features2]))
            test_score = scaler2.transform(test_score)
            RMSE = np.zeros(test_score.shape[0])
            pred = Output.predict(test_score)
            for i in range(test_score.shape[0]):
                RMSE[i] = np.sqrt(metrics.mean_squared_error(pred[i], test_score[i]))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, RMSE)
            indices = np.where(fpr >= 0.01)
            index = np.min(indices)
            soglia = thresholds[index]
            labels = np.zeros(RMSE.shape[0])
            for i in range(RMSE.shape[0]):
                if RMSE[i] < soglia:
                    labels[i] = 0
                else:
                    labels[i] = 1
            for_plot = {'y_true': y_test, 'y_pred': labels, 'RMSE': RMSE}
            plot_df = pd.DataFrame.from_dict(for_plot)

            plot_df.to_csv(outdir + "/results.csv")


if __name__ == '__main__':

    args_names = ['<BASEDIR>', '<MODE>']
    optargs_names = ['<OUTDIR=None>', '<EPOCHS=[1, 5, 10]>', '<ATTACK=All>', '<USE_DUP=False>', '<STRATIFY=True>',
                     '<ONLY_METADATA=False>', '<NFEATS=79>', '<FEATURE_MODE=max>', '<EARLY_STOPPING=False>']

    if len(sys.argv) <= len(args_names):
        print('Usage:', sys.argv[0], ' '.join(args_names), '{', ' '.join(optargs_names), '}')
        exit()

    basedir = sys.argv[1]
    mode = sys.argv[2]
    outdir = sys.argv[3] if len(sys.argv) > 3 else None
    epochs = literal_eval(sys.argv[4]) if len(sys.argv) > 4 else [1, 5, 10]
    attack = sys.argv[5] if len(sys.argv) > 5 else 'all'
    use_dup = literal_eval(sys.argv[6]) if len(sys.argv) > 6 else False
    stratify = literal_eval(sys.argv[7]) if len(sys.argv) > 7 else True
    only_metadata = literal_eval(sys.argv[8]) if len(sys.argv) > 8 else False
    nfeats = literal_eval(sys.argv[9]) if len(sys.argv) > 9 else 79
    feature_mode = sys.argv[10] if len(sys.argv) > 10 else 'max'
    early_stopping = literal_eval(sys.argv[11]) if len(sys.argv) > 11 else False

    if not isinstance(epochs, list):
        epochs = [epochs]

    if attack.lower() == 'all':
        attack = None

    if mode == 'feats':
        Kitsune_Features(basedir=basedir, epochs=epochs, nfeats=nfeats, feature_mode=feature_mode, stratify=stratify,
                         early_stopping=early_stopping, only_metadata=only_metadata, outdir=outdir, attack=attack,
                         use_dup=use_dup)
    elif mode == 'pois':
        Kitsune_Poisoning(basedir=basedir, use_dup=use_dup, only_metadata=only_metadata, attack=attack,
                          outdir=outdir, stratify=stratify, early_stopping=early_stopping, nfeats=nfeats)
