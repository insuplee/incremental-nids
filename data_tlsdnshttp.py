#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# Jan 2020


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd
from sklearn.feature_selection import SelectFdr, chi2  # 200103 (Fri)
from sklearn.model_selection import train_test_split  # 200103 (Fri)

# ----------------------------------------------------------------------------------
# Features
# ---------------------------------
# NOTICE : select features manually (assign to variable "g_selected_features_list")
g_possible_features_list = [
    "src_port", "dst_port", "ciphersuites", "ciphersuites_candidate_length",
    "extension_support_group", "cert_startdate", "cert_enddate", "certificate_length",
    "filename", "session_key", "src_ip", "dst_ip", "extension_server_name",
    "ssl_handshake_version", "certificate", "DNS_qry_name", "DNS_resp_name", "DNS_CNAME",
    "http_request_version", "http_accept_language", "http_content_type", "http_content_length",
    "http_server", "http_user_agent", "interarrival_time", "TCP_segment_len", "frame_time",
    "TCP_segment_payload", "cipersuites_candidate", "extension_length", "cert_isExpired"
]

g_selected_features_list = [
    "src_port", "dst_port", "ciphersuites", "ciphersuites_candidate_length",
    "extension_support_group", "cert_startdate", "cert_enddate", "certificate_length",
    "filename", "session_key", "src_ip", "dst_ip", "extension_server_name",
    "ssl_handshake_version", "certificate", "DNS_qry_name", "DNS_resp_name", "DNS_CNAME",
    "http_request_version", "http_accept_language", "http_content_type", "http_content_length",
    "http_server", "http_user_agent", "interarrival_time", "TCP_segment_len", "frame_time",
    "TCP_segment_payload", "cipersuites_candidate", "extension_length", "cert_isExpired"
]

# g_selected_features_list = g_possible_features_list[:]


def extract_feature(all_feature_df, selected_features):
    extracted_list = list()
    all_feature_list = list(all_feature_df)
    for s_f in selected_features:
        for feature in all_feature_list:
            # if s_f in feature :  # NOTICE
            if s_f in feature and "{}_".format(s_f) not in feature:
                extracted_list.append(feature)
                # print("[!] {} append ".format(feature))
    return extracted_list


# ----------------------------------------------------------------------------------
# Data : TLS / DNS / HTTP
# ---------------------------------
class TlsDnsHttp:
    def __init__(self, bin_clf=True):
        X, y = load_data_tdh(bin_clf=bin_clf)

        self.X, self.y = X, y
        self.bin_clf = bin_clf

    def get_data(self):
        return self.X, self.y

    def get_split_data(self, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test

    # NOTICE: after SelectFdr, type conversion(DataFrame -> ndarray)
    def fdr_get_data(self, fdr_alpha=0.001):
        # print(type(self.X))
        # print(len(self.X.columns))
        fdr_X = SelectFdr(chi2, alpha=fdr_alpha).fit_transform(self.X, self.y)
        # print(type(fdr_X))
        # print(fdr_X.shape[1])
        return fdr_X, self.y

    def fdr_get_split_data(self, fdr_alpha=0.001, test_size=0.3, random_state=42):
        fdr_X, y = self.fdr_get_data(fdr_alpha=fdr_alpha)
        fdr_X_train, fdr_X_test, y_train, y_test = train_test_split(fdr_X, y, test_size=test_size,
                                                                    random_state=random_state)
        return fdr_X_train, fdr_X_test, y_train, y_test


def load_data_tdh(dir_path='datasets/tlsdnshttp', bin_clf=True):
    # file_name = "fe_tlsdnshttp.csv"
    file_name = "fe_tlsdnshttp_rs42.csv"
    train_test_file = os.path.join(dir_path, file_name)

    df = pd.read_csv(train_test_file)
    df = df.mask(df < 0, 0)  # NOTICE : Process values (under zero)

    X = df.iloc[:, 1:]

    # NOTICE : Feature Selection
    # NOTICE : DEBUG
    print("[!] {} Features EXTRACTED such as {}".format(len(g_selected_features_list), g_selected_features_list[:]))
    extracted = extract_feature(X.columns, g_selected_features_list)
    X = X[extracted]

    y = df.iloc[:, 0]  # NOTICE : Label is at column 0

    # prepare for binary classification
    if bin_clf:
        y = y.apply(lambda x: int(x))
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        # print(le.classes_)  # use when needed

    return X, y


if __name__ == "__main__":
    print("Hello TLS-DNS-HTTP")
    tdh = TlsDnsHttp()
    f_X, f_y = tdh.fdr_get_data()

