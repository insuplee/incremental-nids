#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# Jan 2020


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFdr, chi2  # NOTICE : 테스트
from sklearn.model_selection import train_test_split  # NOTICE : 테스트
from sklearn.preprocessing import MinMaxScaler

from experiment_setting import label_encoding_dict

warnings.filterwarnings('ignore')


class NslKddData:
    def __init__(self, bin_clf=True):
        X_train, X_test, y_train, y_test = load_data_nslkdd(bin_clf=bin_clf)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        # NOTICE : in NSL-KDD, trainset and testset files are distinct -> we combine them
        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)
        self.X, self.y = X, y

        if X_train.shape[1] == X_test.shape[1]:  # check : should be same
            self.dim = X_train.shape[1]
        else:
            print("[!] Error occured. X_train and X_test have different feature num")
            raise Exception

        self.bin_clf = bin_clf

    def get_data(self):
        return self.X, self.y

    def get_split_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fdr_get_data(self, fdr_alpha=0.001):
        fdr_X = SelectFdr(chi2, alpha=fdr_alpha).fit_transform(self.X, self.y)
        return fdr_X, self.y

    def fdr_get_split_data(self, fdr_alpha=0.001, test_size=0.3, random_state=42):
        fdr_X, y = self.fdr_get_data(fdr_alpha=fdr_alpha)
        fdr_X_train, fdr_X_test, y_train, y_test = train_test_split(fdr_X, y, test_size=test_size,
                                                                    random_state=random_state)
        return fdr_X_train, fdr_X_test, y_train, y_test


# NOTICE : "KDDTrain+.txt", "KDDTrain+_20Percent.txt", "KDDTest+.txt", "KDDTest-21.txt"
def load_data_nslkdd(dir_path='datasets/nsl-kdd', bin_clf=True):
    nominal_idx = [1, 2, 3]
    binary_idx = [6, 11, 13, 14, 20, 21]
    numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx))

    atk_info_txt = 'datasets/training_attack_types.txt'
    train_file = os.path.join(dir_path, 'KDDTrain+.txt')  # dim 118
    # train_file = os.path.join(dir_path, 'KDDTrain+_20Percent.txt')  # dim 115
    test_file = os.path.join(dir_path, 'KDDTest+.txt')

    feature_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                     'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                     'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                     'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                     'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                     'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                     'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                     'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                     'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                     'attack_type', 'success_pred']

    # Differentiating between nominal, binary, and numeric features
    col_names = np.array(feature_names)
    nominal_cols = list(col_names[nominal_idx])
    binary_cols = list(col_names[binary_idx])
    numeric_cols = list(col_names[numeric_idx])

    # training_attack_types.txt maps each of the 22 different attacks to 1 of 4 categories
    c = defaultdict(list)
    c['benign'].append('normal')
    with open(atk_info_txt, 'r') as rf:
        for line in rf.readlines():
            attack, cat = line.strip().split(' ')
            c[cat].append(attack)
    attack_mapping = dict((v, k) for k in c for v in c[k])

    train_df = pd.read_csv(train_file, names=feature_names)
    train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping[x])
    train_df.drop(['success_pred'], axis=1, inplace=True)  # NOTICE : feature drop

    test_df = pd.read_csv(test_file, names=feature_names)
    test_df['attack_category'] = test_df['attack_type'].map(lambda x: attack_mapping[x])
    test_df.drop(['success_pred'], axis=1, inplace=True)  # NOTICE : feature drop

    # Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0
    train_df['su_attempted'].replace(2, 0, inplace=True)
    test_df['su_attempted'].replace(2, 0, inplace=True)

    # Now, that's not a very useful feature - let's drop it from the dataset
    train_df.drop('num_outbound_cmds', axis=1, inplace=True)
    test_df.drop('num_outbound_cmds', axis=1, inplace=True)
    numeric_cols.remove('num_outbound_cmds')

    # NOTICE : Data preparation
    X_train_raw = train_df.drop(['attack_category', 'attack_type'], axis=1)
    y_train = train_df['attack_category']
    X_test_raw = test_df.drop(['attack_category', 'attack_type'], axis=1)
    y_test = test_df['attack_category']

    combined_df_raw = pd.concat([X_train_raw, X_test_raw])

    # NOTICE: dimension 40->115
    combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

    # return combined_df.shape[1]  # NOTICE : calculate dimenstion
    idx = len(X_train_raw)
    X_train = combined_df[:idx]
    X_test = combined_df[idx:]

    # Store dummy variable feature names
    dummy_variables = list(set(X_train) - set(combined_df_raw))

    # NOTICE : one feature - duration
    durations = X_train['duration'].values.reshape(-1, 1)
    min_max_scaler = MinMaxScaler().fit(durations)
    min_max_scaled_durations = min_max_scaler.transform(durations)
    pd.Series(min_max_scaled_durations.flatten()).describe()

    # NOTICE : all the numeric column => min-max scaler
    min_max_scaler = MinMaxScaler().fit(X_train[numeric_cols])
    X_train[numeric_cols] = min_max_scaler.transform(X_train[numeric_cols])
    X_test[numeric_cols] = min_max_scaler.transform(X_test[numeric_cols])

    # NOTICE : Label encoding
    y_train = y_train.apply(encoding_label)
    y_test = y_test.apply(encoding_label)

    return X_train, X_test, y_train, y_test


def encoding_label(cat):
    if cat in label_encoding_dict.keys():
        return label_encoding_dict[cat]
    else:
        print("[!] Error in encoding_label")
        print("cat is {} and keys are {}".format(cat, label_encoding_dict.keys()))
        raise Exception


if __name__ == "__main__":
    print("Hello NSL-KDD")
    n = NslKddData(bin_clf=True)

    fdr_X, fdr_y = n.fdr_get_data(fdr_alpha=0.1)
    print("[!] DEBUG : shape is {}".format(fdr_X.shape))
    fdr_X, fdr_y = n.fdr_get_data(fdr_alpha=0.01)
    print("[!] DEBUG : shape is {}".format(fdr_X.shape))
    fdr_X, fdr_y = n.fdr_get_data(fdr_alpha=0.001)
    print("[!] DEBUG : shape is {}".format(fdr_X.shape))
