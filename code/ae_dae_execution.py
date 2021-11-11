import os
import statistics
import sys
import timeit
from ast import literal_eval
from time import process_time as time

import numpy as np
import pandas as pd
from common_lib import *
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers


def create_autoencoder(x_train, compression_ratio=.75):
    input_dim = x_train.shape[1]
    encoding_dim = round(input_dim * compression_ratio)

    input_data = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_data)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = keras.Model(input_data, decoded)

    opt = keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    return autoencoder


def create_deep_autoencoder(x_train, compression_ratios=[.75, .5, .33, .25]):
    input_dim = x_train.shape[1]
    dims = []
    for compression_ratio in compression_ratios:
        dims.append(round(input_dim * compression_ratio))

    input_data = keras.Input(shape=(input_dim,))

    dense = input_data
    for dim in dims:
        dense = layers.Dense(dim, activation='relu')(dense)
    for dim in dims[:-1][::-1]:
        dense = layers.Dense(dim, activation='relu')(dense)
    decoded = layers.Dense(input_dim, activation='sigmoid')(dense)

    deep_autoencoder = keras.Model(input_data, decoded)

    opt = keras.optimizers.Adam(learning_rate=0.001)
    deep_autoencoder.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    return deep_autoencoder


def Detector_Features(basedir='.', outdir=None, epochs=[1], model_name='sae', attack=None, use_dup=False, stratify=True,
                      only_metadata=False, nfeats=71, feature_mode='max', early_stopping=False):
    train_data_n, basedir = load_dataset_n_basedir(basedir, attack, use_dup)

    if outdir is not None:
        basedir = '%s/%s' % (outdir, basedir.replace('../', ''))

    model_fullname = 'Autoencoder' if model_name == 'sae' else 'Deep_Autoencoder'
    feature_names = np.array(train_data_n.columns)[:nfeats]

    train_data_n = np.array(train_data_n)
    train_data_labels = train_data_n[:, -2]
    train_data_stratify = train_data_n[:, -1]
    train_data_stratify = np.array([s if l == 0 else s + 100 for l, s in zip(train_data_labels, train_data_stratify)])
    train_data_n = train_data_n[:, :nfeats]

    for fold, (train_index, test_index) in enumerate(dataset_split(
            len(train_data_n), train_data_stratify if stratify else None, random_state=0
    )):
        train_index = train_index.astype('int32')
        test_index = test_index.astype('int32')
        x_train = train_data_n[train_index]
        y_train = train_data_labels[train_index]
        x_test = train_data_n[test_index]
        y_test = train_data_labels[test_index]
        y_test = y_test.astype('int')

        # Filtering out malicious traffic
        benign_index = (y_train == 0)
        x_train = x_train[benign_index]
        y_train = y_train[benign_index]

        selected_features = n_features_selection(x_train, y_train, feature_mode=feature_mode)

        # Training and Test against number of selected features
        for n_features, features in selected_features.items():
            for epoch in epochs:
                # Building of train and test sets given features
                training_set = x_train[:, features]
                test_set = x_test[:, features]
                # Dataset normalization
                scaler = MinMaxScaler(feature_range=(0, 1))
                train = scaler.fit_transform(training_set)
                test = scaler.transform(test_set)

                outdir = '%s/Training/%s/fix/Epochs%s/Fold%s' % (basedir, model_fullname, epoch, fold + 1)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                # Storing metadata
                df_meta_tr = pd.DataFrame(columns=['Malign', 'Class'], index=range(len(x_train)))
                df_meta_te = pd.DataFrame(columns=['Malign', 'Class'], index=range(len(x_test)))
                df_sel_feature_names = pd.DataFrame(columns=feature_names[features])
                df_meta_tr['Malign'] = y_train
                df_meta_tr['Class'] = [v if v < 100 else v - 100 for v in
                                       train_data_stratify[train_index][benign_index]]
                df_meta_te['Malign'] = y_test
                df_meta_te['Class'] = [v if v < 100 else v - 100 for v in train_data_stratify[test_index]]

                my_callbacks = get_callbacks(outdir, '_%s' % n_features, early_stopping)

                # Model Creation and Training
                my_model = create_autoencoder(train) if model_name == 'sae' else create_deep_autoencoder(train)

                # Procedure to evaluate model inference time
                df_params_inference_info = pd.DataFrame(
                    columns=['Model Name', 'Trainable Parameters', 'Inference Time [s]']
                )
                n_dummy = 500
                X = np.random.random((n_dummy, n_features))
                ts = []
                for _ in range(10):
                    t = time()
                    my_model.predict(X)
                    t = time() - t
                    ts.append(t / n_dummy)
                t = np.median(ts)
                df_params_inference_info = df_params_inference_info.append(
                    {
                        'Model Name': model_fullname,
                        'Trainable Parameters': sum([np.prod(v.shape) for v in my_model.trainable_variables]),
                        'Inference Time [s]': t
                    }, ignore_index=True
                )

                df_meta_tr.to_csv('%s/training_labels_%s.csv' % (outdir, n_features), index=False)
                df_meta_te.to_csv('%s/test_labels_%s.csv' % (outdir, n_features), index=False)
                df_sel_feature_names.to_csv('%s/selected_features_%s.csv' % (outdir, n_features), index=False)
                df_params_inference_info.to_csv('%s/params_n_inference_%s.csv' % (outdir, n_features), index=False)

                if only_metadata:
                    continue

                history = my_model.fit(train, train, epochs=epoch, batch_size=32, shuffle=True, callbacks=my_callbacks,
                                       validation_data=(test[(y_test == 0)], test[(y_test == 0)]))

                # Training stats storing in csv
                history_df = pd.DataFrame(history.history)
                hist_csv_file = '%s/history_%s.csv' % (outdir, n_features)
                with open(hist_csv_file, mode='w') as f:
                    history_df.to_csv(f)

                # Predict on train set
                train_pred = my_model.predict(train)

                # Per-biflow RMSE computation on predicted features
                train_rmse = list()
                for i in range(len(train)):
                    train_rmse.append(np.sqrt(mean_squared_error(train[i], train_pred[i])))

                # RMSE stats
                mean = statistics.mean(train_rmse)
                dev = statistics.stdev(train_rmse)

                # Predict on test set
                test_pred = my_model.predict(test)
                test_rmse = list()
                for i in range(len(test)):
                    test_rmse.append(np.sqrt(mean_squared_error(test[i], test_pred[i])))

                # Computation of fpr and tpr in order to determine the 1% FPR threshold for the F1 Score
                fpr, tpr, thresholds = metrics.roc_curve(y_test, test_rmse)
                indices = np.where(fpr >= 0.01)
                index = np.min(indices)
                soglia = thresholds[index]
                label_pred = list()
                for i in range(len(test)):
                    if test_rmse[i] >= soglia:
                        label_pred.append(1)
                    else:
                        label_pred.append(0)

                f1_score = metrics.f1_score(y_test, label_pred, average='macro')

                # Storing info in csv
                stats = {'mean': mean, 'dev': dev, 'threshold': soglia, 'f1_score': f1_score}
                stats_df = pd.DataFrame.from_dict(stats, orient='index')
                stats_df.to_csv('%s/stats_%s.csv' % (outdir, n_features))

                # Storing results in csv
                for_plot = {'y_true': y_test, 'y_pred': label_pred, 'RMSE': test_rmse}
                plot_df = pd.DataFrame.from_dict(for_plot)
                plot_df.to_csv('%s/plot_%s.csv' % (outdir, n_features))


def Detector_Poisoning(basedir='.', outdir=None, model_name='sae', attack=None, use_dup=False, stratify=True,
                       only_metadata=False, nfeats=71, early_stopping=False):
    train_data_n, basedir = load_dataset_n_basedir(basedir, attack, use_dup)

    if outdir is not None:
        basedir = '%s/%s' % (outdir, basedir.replace('../', ''))

    model_fullname = 'Autoencoder' if model_name == 'sae' else 'Deep_Autoencoder'
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

            outdir = '%s/Training/%s/Poisoning/%s%%/Fold%s/' % (
                basedir, model_fullname, percent_poisoning, iteration + 1
            )
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

            scaler = MinMaxScaler(feature_range=(0, 1))
            training_features = scaler.fit_transform(x_train)
            test_features = scaler.transform(x_test)

            my_callbacks = get_callbacks(outdir, early_stopping=early_stopping)

            # Model Creation and Training
            my_model = create_autoencoder(x_train) if model_name == 'sae' else create_deep_autoencoder(x_train)

            history = my_model.fit(training_features, training_features, epochs=100, batch_size=32, shuffle=True,
                                   callbacks=my_callbacks)

            # Training stats storing in csv
            history_df = pd.DataFrame(history.history)
            hist_csv_file = '%s/history.csv' % outdir
            with open(hist_csv_file, mode='w') as f:
                history_df.to_csv(f)

            # Predict on train set
            train_pred = my_model.predict(training_features)

            # Calcolo RMSE sulle features predette sui singoli biflussi
            train_rmse = list()
            for i in range(len(training_features)):
                train_rmse.append(np.sqrt(mean_squared_error(training_features[i], train_pred[i])))

            # RMSE stats
            mean = statistics.mean(train_rmse)
            dev = statistics.stdev(train_rmse)

            # Predict on test set
            test_pred = my_model.predict(test_features)
            test_rmse = list()
            for i in range(len(test_features)):
                test_rmse.append(np.sqrt(mean_squared_error(test_features[i], test_pred[i])))

            # Computation of fpr and tpr in order to determine the 1% FPR threshold for the F1 Score
            fpr, tpr, thresholds = metrics.roc_curve(y_test, test_rmse)
            indices = np.where(fpr >= 0.01)
            index = np.min(indices)
            soglia = thresholds[index]
            label_pred = list()
            for i in range(len(test_features)):
                if test_rmse[i] >= soglia:
                    label_pred.append(1)
                else:
                    label_pred.append(0)

            f1_score = metrics.f1_score(y_test, label_pred, average='macro')

            # Storing info in csv
            stats = {'mean': mean, 'dev': dev, 'threshold': soglia, 'f1_score': f1_score}
            stats_df = pd.DataFrame.from_dict(stats, orient='index')
            stats_df.to_csv('%s/stats.csv' % outdir)

            # Storing results in csv
            for_plot = {'y_true': y_test, 'y_pred': label_pred, 'RMSE': test_rmse}
            plot_df = pd.DataFrame.from_dict(for_plot)
            plot_df.to_csv('%s/plot.csv' % outdir)


if __name__ == '__main__':

    args_names = ['<BASEDIR>', '<MODE>', '<MODEL_NAME>']
    optargs_names = ['<OUTDIR=None>', '<EPOCHS=[1, 5, 10]>', '<ATTACK=All>', '<USE_DUP=False>', '<STRATIFY=True>',
                     '<ONLY_METADATA=False>', '<NFEATS=79>', '<FEATURE_MODE=max>', '<EARLY_STOPPING=False>']

    if len(sys.argv) <= len(args_names):
        print('Usage:', sys.argv[0], ' '.join(args_names), '{', ' '.join(optargs_names), '}')
        exit()

    basedir = sys.argv[1]
    mode = sys.argv[2]
    model_name = sys.argv[3]
    outdir = sys.argv[4] if len(sys.argv) > 4 else None
    epochs = literal_eval(sys.argv[5]) if len(sys.argv) > 5 else [1, 5, 10]
    attack = sys.argv[6] if len(sys.argv) > 6 else 'all'
    use_dup = literal_eval(sys.argv[7]) if len(sys.argv) > 7 else False
    stratify = literal_eval(sys.argv[8]) if len(sys.argv) > 8 else True
    only_metadata = literal_eval(sys.argv[9]) if len(sys.argv) > 9 else False
    nfeats = literal_eval(sys.argv[10]) if len(sys.argv) > 10 else 79
    feature_mode = sys.argv[11] if len(sys.argv) > 11 else 'max'
    early_stopping = literal_eval(sys.argv[12]) if len(sys.argv) > 12 else False

    if not isinstance(epochs, list):
        epochs = [epochs]

    if attack.lower() == 'all':
        attack = None

    if mode == 'feats':
        Detector_Features(basedir=basedir, epochs=epochs, nfeats=nfeats, feature_mode=feature_mode, stratify=stratify,
                          early_stopping=early_stopping, only_metadata=only_metadata, outdir=outdir, attack=attack,
                          use_dup=use_dup, model_name=model_name)
    elif mode == 'pois':
        Detector_Poisoning(basedir=basedir, use_dup=use_dup, only_metadata=only_metadata, attack=attack,
                           outdir=outdir, stratify=stratify, early_stopping=early_stopping, nfeats=nfeats,
                           model_name=model_name)
