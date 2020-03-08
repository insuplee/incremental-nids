#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# Jan 2020


is_online_setting = True
is_bin_clf = True

if is_bin_clf:
    label_encoding_dict = {"benign": 1, "dos": 0, "probe": 0, "u2r": 0, "r2l": 0}
    g_classes = ["attack", "benign"]
    g_encoded_classes = [0, 1]
    g_atk_idx = 0
    g_benign_idx = 1  # NOTICE : g_encoded_classes[g_atk_idx]
else:
    label_encoding_dict = {"benign": 0, "dos": 1, "probe": 2, "u2r": 3, "r2l": 4}
    g_classes = ["benign", "dos", "probe", "u2r", "r2l"]
    g_encoded_classes = [0, 1, 2, 3, 4]

g_algorithms = ["SGD", "MLP", "NB"]  # g_algorithms = ["Perceptron", "SGD", "PA", "NB", "MLP"]
g_algorithms = ["SGD", "PA", "NB", "MLP"]
g_algorithms = ["SGD", "PA", "NB"]


g_fdr_alpha_list = [0.001]  # g_fdr_alpha_list = [0.001, 0.01, 0.1, 1]
g_n_iter = 5
g_test_size = 0.3

g_fig_dir = "./figures"

g_figsize = (16, 10) #  (8, 5)

g_title_fontsize = 40  # 15
g_xlabel_fontsize = 30  # 15
g_y_label_fontsize = 30  # 14
g_annotate_font_size = 25
g_x_ticks_font_size = 40
g_y_ticks_font_size = 30
g_legend_font_size = 30

g_draw_linewidth = 3
g_draw_markersize = 15