#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# Mar 2020

import os

import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def print_ref_sites():
    site_dict = {
        "parm_opt": "https://scikit-learn.org/stable/modules/grid_search.html",
        "incremental_concept": "https://tensorflow.blog/%ED%95%B8%EC%A6%88%EC%98%A8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-1%EC%9E%A5-2%EC%9E%A5/1-3-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%8B%9C%EC%8A%A4%ED%85%9C%EC%9D%98-%EC%A2%85%EB%A5%98/"
    }

    print("== all site information ==")
    for site in site_dict:
        print(site, site_dict[site])


def main():
    print("################################################################################")
    print("[!] test for INFOCOM Poster")
    print("[!] Poster Abstract: Encrypted Malware Traffic Detection using Incremental Learning ")
    print("################################################################################")

    clf_list = [SGDClassifier(), GaussianNB(), MLPClassifier(), PassiveAggressiveClassifier()]

    for clf in clf_list:
        print(clf.__class__.__name__)
        print(clf.get_params())


if __name__ == "__main__":
    print("HELLO MAIN")
    main()
    print_ref_sites()