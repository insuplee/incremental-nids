#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# April 2020

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_tlsdnshttp import TlsDnsHttp

def get_colors(t):
    color_dict = {0: "#0392cf", 1: "#7bc043", 2: "#ee4035"}
    colors = list()
    for i in range(len(t)):
        colors.append(color_dict[t[i]])
    return np.array(colors)

def main():
    tdh = TlsDnsHttp()
    X, y = tdh.get_data()

    do_plot = False
    if do_plot:
        colors = get_colors(y)
        pd.plotting.scatter_matrix(X, color=colors, diagonal='kde')

    print(X.shape)

    print(X['interarrival_time'])
    print(X['frame_time'])
    return
    for i in range(4):
        start_idx = i*8
        last_idx = (i+1)*8
        if i == 3:
            last_idx -= 1

        X_temp = X.iloc[:, list(range(start_idx, last_idx))]
        X_temp.boxplot()
        plt.show()

    with open(os.path.join("result_feature-engineering", "describe.txt"), "wt") as wf:
        wf.write(str(X.describe()))
    #print(X.describe())


if __name__ == "__main__":
    print("HELLO MAIN")
    start_time = time.time()
    main()
    print("[!] time {:.3f} seconds".format(time.time() - start_time))
    print("END")
