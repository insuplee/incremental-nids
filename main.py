#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# Jan 2020

import os

import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier

from data_nslkdd import NslKddData
from data_tlsdnshttp import TlsDnsHttp
from experiment_setting import is_bin_clf, g_encoded_classes, g_atk_idx, g_benign_idx, g_algorithms, g_figsize, \
    g_fdr_alpha_list, g_n_iter, g_test_size, g_xlabel_fontsize, g_y_label_fontsize, g_fig_dir, g_annotate_font_size, \
    g_x_ticks_font_size, g_y_ticks_font_size, g_legend_font_size, g_draw_linewidth, g_draw_markersize  # NOTICE


# ----------------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------
# Calculate results (acc, pre, rec, f1, fpr) using confusion matrix
def get_performance_dict(clf, X_test, y_test, bin_clf=True):
    y_pred = clf.predict(X_test)

    if bin_clf:
        info_dict = dict()

        info_dict["common"] = dict()
        info_dict["attack"] = dict()
        info_dict["benign"] = dict()

        atk_encoded = g_encoded_classes[g_atk_idx]
        benign_encoded = g_encoded_classes[g_benign_idx]

        # NOTICE : If y_test == y_pred, cm -> 1by1 (Error occur)
        # TODO : confusion matrix visualization
        cm = confusion_matrix(y_test, y_pred)

        tn = cm[0, 0]  # True-Negative
        fp = cm[0, 1]  # False-Positive
        fn = cm[1, 0]  # False-Negative
        tp = cm[1, 1]  # True-Positive

        cm_acc = (tp + tn) / (tp + tn + fp + fn)

        b_cm_pre = tp / (tp + fp)
        b_cm_rec = tp / (tp + fn)  # TPR
        b_cm_f1 = 2 * (b_cm_pre * b_cm_rec) / (b_cm_pre + b_cm_rec)
        b_cm_fpr = fp / (tn + fp)

        a_cm_pre = tn / (tn + fn)
        a_cm_rec = tn / (tn + fp)  # TPR
        a_cm_f1 = 2 * (a_cm_pre * a_cm_rec) / (a_cm_pre + a_cm_rec)
        a_cm_fpr = fn / (tp + fn)

        # extract information
        info_dict["common"]["accuracy"] = float("{:.3f}".format(cm_acc))
        info_dict["common"]["total_support"] = len(y_test)

        info_dict["attack"]["encoded_label"] = g_encoded_classes[g_atk_idx]
        info_dict["attack"]["precision"] = float("{:.3f}".format(a_cm_pre))
        info_dict["attack"]["recall"] = float("{:.3f}".format(a_cm_rec))
        info_dict["attack"]["f1_score"] = float("{:.3f}".format(a_cm_f1))
        info_dict["attack"]["fpr"] = float("{:.3f}".format(a_cm_fpr))
        info_dict["attack"]["support"] = len([k for k in y_test if k == atk_encoded])

        info_dict["benign"]["encoded_label"] = g_encoded_classes[g_benign_idx]
        info_dict["benign"]["precision"] = float("{:.3f}".format(b_cm_pre))
        info_dict["benign"]["recall"] = float("{:.3f}".format(b_cm_rec))
        info_dict["benign"]["f1_score"] = float("{:.3f}".format(b_cm_f1))
        info_dict["benign"]["fpr"] = float("{:.3f}".format(b_cm_fpr))
        info_dict["benign"]["support"] = len([k for k in y_test if k == benign_encoded])

    else:
        print("[!] must be binary class!")
        raise Exception
    return info_dict


# [1] Main Evaluation setting of Incremental Learning
# offline evaluation : explicitly split all data into a training set and a testing set
# ref : [2017] Viktor Losing_incremental on-line learning : a review and comparison of state of the art algorithms
def offline_setting(inc_clf, X_train, X_test, y_train, y_test, n_iter):
    is_last_minibatch = False
    train_data_size = X_train.shape[0]
    minibatch_size = train_data_size // n_iter
    performance_dict_list = []

    for i in range(0, train_data_size, minibatch_size):
        # NOTICE: To prevent the last block data is too small.. process the last block data beforehand
        if i + 2 * minibatch_size >= train_data_size:
            X_train_mini = X_train[i:]
            y_train_mini = y_train[i:]
            is_last_minibatch = True
        else:
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]

        # NOTICE : classes of partial_fit?
        inc_clf.partial_fit(X_train_mini, y_train_mini, classes=g_encoded_classes)

        if is_last_minibatch:
            break

    performance_dict_list.append(get_performance_dict(inc_clf, X_test, y_test))
    return performance_dict_list


# [2] Main Evaluation setting of Incremental Learning
# online evaluation : does NOT split all data. testing first, then train.
# ref : [2017] Viktor Losing_incremental on-line learning : a review and comparison of state of the art algorithms
def online_setting(inc_clf, X, y, n_iter):
    is_first_minibatch = True
    is_last_minibatch = False
    train_data_size = X.shape[0]
    minibatch_size = train_data_size // n_iter
    performance_dict_list = []

    for i in range(0, train_data_size, minibatch_size):
        # NOTICE: To prevent the last block data is too small.. process the last block data beforehand
        if i + 2 * minibatch_size >= train_data_size:
            X_mini = X[i:]
            y_mini = y[i:]
            is_last_minibatch = True
        else:
            X_mini = X[i:i + minibatch_size]
            y_mini = y[i:i + minibatch_size]

        if is_first_minibatch:  # skip the test in first model
            is_first_minibatch = False
            # print("i is {} and minibatch test skipped!".format(i))
        else:
            performance_dict_list.append(get_performance_dict(inc_clf, X_mini, y_mini))
            # print("i is {} and minibatch test done!".format(i))

        # NOTICE : classes of partial_Fit?
        # print("i is {} and model train!".format(i))
        inc_clf.partial_fit(X_mini, y_mini, classes=g_encoded_classes)

        if is_last_minibatch:
            break

    return performance_dict_list


# After step of off/online_setting evaluation
def get_output(dict_list):
    acc, tospt = "accuracy", "total_support"  # NOTICE: tospt != ts (semantically)
    pre, rec, f1, fpr, spt = "precision", "recall", "f1_score", "fpr", "support"

    ta, ts = "total_accuracy", "total_support"
    tp, tr, tf1, tfpr = "total_precision", "total_recall", "total_f1_score", "total_fpr"

    output_dict = dict()
    output_dict["common"] = {ta: 0.0, ts: 0}
    output_dict["attack"] = {tp: 0.0, tr: 0.0, tf1: 0.0, tfpr: 0.0, ts: 0}
    output_dict["benign"] = {tp: 0.0, tr: 0.0, tf1: 0.0, tfpr: 0.0, ts: 0}

    for tmp_dict in dict_list:
        output_dict["common"][ta] += tmp_dict["common"][acc] * tmp_dict["common"][tospt]
        output_dict["common"][ts] += tmp_dict["common"][tospt]

        output_dict["attack"][tp] += tmp_dict["attack"][pre] * tmp_dict["attack"][spt]
        output_dict["attack"][tr] += tmp_dict["attack"][rec] * tmp_dict["attack"][spt]
        output_dict["attack"][tf1] += tmp_dict["attack"][f1] * tmp_dict["attack"][spt]
        output_dict["attack"][tfpr] += tmp_dict["attack"][fpr] * tmp_dict["attack"][spt]
        output_dict["attack"][ts] += tmp_dict["attack"][spt]

        output_dict["benign"][tp] += tmp_dict["benign"][pre] * tmp_dict["benign"][spt]
        output_dict["benign"][tr] += tmp_dict["benign"][rec] * tmp_dict["benign"][spt]
        output_dict["benign"][tf1] += tmp_dict["benign"][f1] * tmp_dict["benign"][spt]
        output_dict["benign"][tfpr] += tmp_dict["benign"][fpr] * tmp_dict["benign"][spt]
        output_dict["benign"][ts] += tmp_dict["benign"][spt]

    res_dict = dict()
    res_dict["common"], res_dict["attack"], res_dict["benign"] = dict(), dict(), dict()

    res_dict["common"]["avg_accuracy"] = output_dict["common"][ta] / output_dict["common"][ts]
    res_dict["common"]["total_support"] = output_dict["common"][ts]

    res_dict["attack"]["avg_precision"] = output_dict["attack"][tp] / output_dict["attack"][ts]
    res_dict["attack"]["avg_recall"] = output_dict["attack"][tr] / output_dict["attack"][ts]
    res_dict["attack"]["avg_f1_score"] = output_dict["attack"][tf1] / output_dict["attack"][ts]
    res_dict["attack"]["avg_fpr"] = output_dict["attack"][tfpr] / output_dict["attack"][ts]
    res_dict["attack"]["total_support"] = output_dict["attack"][ts]

    res_dict["benign"]["avg_precision"] = output_dict["benign"][tp] / output_dict["benign"][ts]
    res_dict["benign"]["avg_recall"] = output_dict["benign"][tr] / output_dict["benign"][ts]
    res_dict["benign"]["avg_f1_score"] = output_dict["benign"][tf1] / output_dict["benign"][ts]
    res_dict["benign"]["avg_fpr"] = output_dict["benign"][tfpr] / output_dict["benign"][ts]
    res_dict["benign"]["total_support"] = output_dict["benign"][ts]

    return res_dict


# ----------------------------------------------------------------------------------
# Drawing graph functions
# ---------------------------------
def draw_bar_graph(res_dict_list, do_eval_online=True, metric="accuracy", desc="", do_kisti=False):
    info_list = res_dict_list[0].split("_")
    fdr_info = float(info_list[0].split("-")[-1])
    if do_eval_online:
        title = "online"
    else:
        title = "offline"

    if do_kisti:
        print("[!] DRAW_RES : KISTI...")
    else:
        print("[!] DRAW_RES : NSL-KDD...")

    algo_name_list = []
    fdr_acc_list = []
    for res_dict in res_dict_list:
        res_info = res_dict.split("_")
        res_fdr = float(res_info[0].split("-")[-1])
        if res_fdr != fdr_info:
            print("[!] fdr info error")
            raise Exception
        res_algo = res_info[1]
        res_fdr_acc = float(res_info[2].split("-")[-1])

        algo_name_list.append(res_algo)
        fdr_acc_list.append(res_fdr_acc)

    plt.figure(figsize=g_figsize)
    # plt.title(title, fontsize=g_title_fontsize)
    if do_eval_online:
        plt.bar(algo_name_list, fdr_acc_list, width=0.5, color="green")
    else:
        plt.bar(algo_name_list, fdr_acc_list, width=0.5)
    plt.xticks(list(range(len(g_algorithms))), algo_name_list, fontsize=g_x_ticks_font_size)
    plt.yticks([0.8, 0.85, 0.90, 0.95, 1.0], fontsize=g_y_ticks_font_size)
    plt.xlabel("Incremental Algorithms", fontsize=g_xlabel_fontsize)
    if desc != "":
        plt.ylabel("{} at FDR {} %".format(metric, fdr_info), fontsize=g_y_label_fontsize)
    else:
        plt.ylabel("{} at FDR {} %".format(metric, fdr_info), fontsize=g_y_label_fontsize)

    if do_kisti:
        if metric == "accuracy":
            plt.ylim(0.8, 1.0)
        elif metric == "recall":
            plt.ylim(0.0, 1.0)
    else:  # NSL-KDD
        if metric == "accuracy":
            plt.ylim(0.9, 1.0)
        elif metric == "recall":
            plt.ylim(0.0, 1.0)

    # plt.show()
    plt.savefig("{}\\{}_accuracy.png".format(g_fig_dir, title))



def draw_detailed_online(performance_dict_list, fdr_info=0.001, metric="accuracy", desc="mal-detect",
                         do_kisti=False):
    title = "On-line Setting Process"
    key_list = g_algorithms  # algorithms to draw
    mini_batch_num = len(performance_dict_list["SGD"])

    if do_kisti:
        print("[!] DRAW_RES_DETAILED : KISTI...")
    else:
        print("[!] DRAW_RES_DETAILED : NSL-KDD...")

    plt.figure(figsize=g_figsize)
    # plt.title(title, fontsize=g_title_fontsize)

    key_cnt = 0
    for key in key_list:
        key_cnt += 1
        tmp_list = performance_dict_list[key]
        # annotated_x = [x + 1 for x in range(mini_batch_num)]
        annotated_x = ["#{}".format(x + 1) for x in range(mini_batch_num)]
        annotated_y = tmp_list

        if key_cnt == 1:
            plt.plot(annotated_x, annotated_y, label=key, marker="s", linestyle="-", linewidth=g_draw_linewidth,
                     markersize=g_draw_markersize)  # draw
        elif key_cnt == 2:
            plt.plot(annotated_x, annotated_y, label=key, marker="^", linestyle=":", linewidth=g_draw_linewidth,
                     markersize=g_draw_markersize)  # draw
        elif key_cnt == 3:
            plt.plot(annotated_x, annotated_y, label=key, marker="o", linestyle="--", linewidth=g_draw_linewidth,
                     markersize=g_draw_markersize)  # draw
        else:
            plt.plot(annotated_x, annotated_y, label=key)

        for i, j in zip(annotated_x, annotated_y):
            plt.annotate("{:.3f}".format(j), xy=(i, j), fontsize=g_annotate_font_size)  # value


    # plt.ylim(0.8, 1.0)
    if do_kisti:
        # plt.ylim(0.6, 1.0)
        plt.ylim(0.8, 1.0)
    else:
        plt.ylim(0.95, 1.0)
    # plt.xlabel("mini-batch", fontsize=g_xlabel_fontsize)
    plt.xlabel("Intermediate Model", fontsize=g_xlabel_fontsize)
    plt.ylabel("{} at FDR {} %".format(metric, fdr_info), fontsize=g_y_label_fontsize)
    # plt.xticks(list(k + 1 for k in range(g_n_iter - 1)), annotated_x, fontsize=g_x_ticks_font_size)
    plt.xticks(list(k for k in range(g_n_iter - 1)), annotated_x, fontsize=g_x_ticks_font_size)
    plt.yticks([0.8, 0.85, 0.90, 0.95, 1.0], fontsize=g_y_ticks_font_size)
    plt.legend(fontsize=g_legend_font_size)

    # plt.show()
    plt.savefig("{}\\online_detailed_intermediate.png".format(g_fig_dir))


def calc_real_online(mal_online_acc_list_dict, mal_online_spt_list_dict):
    new_mal_online_acc_list_dict = dict()

    for algo in mal_online_acc_list_dict.keys():
        tmp_acc_list = mal_online_acc_list_dict[algo]
        tmp_spt_list = mal_online_spt_list_dict[algo]
        max_mini_batch_num = len(tmp_acc_list)

        new_mal_online_acc_list_dict[algo] = list()

        for mb_num in range(1, max_mini_batch_num + 1):

            total_acc_sum = 0.0
            total_spt = sum(tmp_spt_list[:mb_num])
            for i in range(mb_num):
                total_acc_sum += tmp_acc_list[i] * tmp_spt_list[i]
            res = total_acc_sum / total_spt
            new_mal_online_acc_list_dict[algo].append(float("{:.3f}".format(res)))

    return new_mal_online_acc_list_dict


def draw_online_learning_curve(performance_dict_list, fdr_info=0.001, metric="accuracy", desc="mal-detect",
                               do_kisti=False):
    key_list = g_algorithms  # algorithms to draw
    mini_batch_num = len(performance_dict_list["SGD"])  # SGD는 임의

    if do_kisti:
        print("[!] DRAW_RES_DETAILED : KISTI...")
    else:
        print("[!] DRAW_RES_DETAILED : NSL-KDD...")

    plt.figure(figsize=g_figsize)
    key_cnt = 0
    for key in key_list:
        key_cnt += 1

        tmp_list = performance_dict_list[key]
        annotated_x = [x + 1 for x in range(mini_batch_num)]
        annotated_y = tmp_list

        if key_cnt == 1:
            plt.plot(annotated_x, annotated_y, label=key, marker="s", linestyle="-", linewidth=g_draw_linewidth,
                     markersize=g_draw_markersize)  # draw
        elif key_cnt == 2:
            plt.plot(annotated_x, annotated_y, label=key, marker="^", linestyle=":", linewidth=g_draw_linewidth,
                     markersize=g_draw_markersize)  # draw
        elif key_cnt == 3:
            plt.plot(annotated_x, annotated_y, label=key, marker="o", linestyle="--", linewidth=g_draw_linewidth,
                     markersize=g_draw_markersize)  # draw
        else:
            plt.plot(annotated_x, annotated_y, label=key)

        for i, j in zip(annotated_x, annotated_y):
            plt.annotate("{:.3f}".format(j), xy=(i, j), fontsize=g_annotate_font_size)  # value

    if do_kisti:
        plt.ylim(0.8, 1.0)
    else:
        plt.ylim(0.95, 1.0)
    plt.xlabel("num of trained samples (chunk)", fontsize=g_xlabel_fontsize)
    plt.ylabel("on-line {} at FDR {} %".format(metric, fdr_info), fontsize=g_y_label_fontsize)
    plt.xticks(list(k + 1 for k in range(g_n_iter - 1)), annotated_x, fontsize=g_x_ticks_font_size)
    plt.yticks([0.8, 0.85, 0.90, 0.95, 1.0], fontsize=g_y_ticks_font_size)
    plt.legend(fontsize=g_legend_font_size)

    # plt.show()
    plt.savefig("{}\\online_learning_curve.png".format(g_fig_dir))


# ----------------------------------------------------------------------------------
# Experimental main functions
# ---------------------------------
def experiment_incremental(bin_clf=True, do_eval_online=True, do_kisti=False):
    all_res_dict_list = []
    mal_all_res_dict_list = []
    benign_all_res_dict_list = []

    # NOTICE : for criteria - recall s
    mal_all_res_dict_list_rec = []

    # NOTICE : for draw_online_learning_curve
    mal_online_acc_dict_list, mal_online_spt_list_dict = dict(), dict()
    for atk in g_algorithms:
        mal_online_acc_dict_list[atk] = list()
        mal_online_spt_list_dict[atk] = list()

    fdr_alpha_list = g_fdr_alpha_list
    n_iter = g_n_iter
    r_s = 42

    # prepare dataset
    if do_kisti:
        n = TlsDnsHttp(bin_clf=bin_clf)
    else:
        n = NslKddData(bin_clf=bin_clf)

    test_cnt = 0
    algorithms = g_algorithms
    for fdr_alpha in fdr_alpha_list:
        for algo in algorithms:  # NOTICE : start
            test_cnt += 1

            if algo == "Multinomial-NB":
                inc_clf = MultinomialNB()
            elif algo == "Bernoulli-NB":
                inc_clf = BernoulliNB()
            elif algo == "Perceptron":
                inc_clf = Perceptron(random_state=r_s)
            elif algo == "SGD":
                inc_clf = SGDClassifier(max_iter=5, random_state=r_s)
            elif algo == "Passive-Aggressive" or algo == "PA":
                inc_clf = PassiveAggressiveClassifier(random_state=r_s)
            elif algo == "MLP":
                inc_clf = MLPClassifier(random_state=r_s)
            elif algo == "Gaussian NB" or algo == "NB":
                inc_clf = GaussianNB()
            else:
                print("[!] Error: Not supported algorithm!!")
                raise Exception

            if do_eval_online:
                X, y = n.fdr_get_data(fdr_alpha=fdr_alpha)
                performance_dict_list = online_setting(inc_clf, X, y, n_iter=n_iter)  # NOTICE: Debugging 200106
                for mini_dict in performance_dict_list:
                    mal_online_acc_dict_list[algo].append(mini_dict["attack"]["precision"])
                    mal_online_spt_list_dict[algo].append(mini_dict["attack"]["support"])  # NOTICE
            else:
                X_train, X_test, y_train, y_test = n.fdr_get_split_data(fdr_alpha, test_size=g_test_size,
                                                                        random_state=r_s)
                performance_dict_list = offline_setting(inc_clf, X_train, X_test, y_train, y_test, n_iter=n_iter)

            # print(performance_dict_list)
            output_dict = get_output(performance_dict_list)

            # NOTICE!!!!
            mal_all_res_dict_list.append(
                "FDR-{}_{}_avg-pre-atk-{:.3f}".format(fdr_alpha, algo, output_dict["attack"]["avg_precision"]))
            benign_all_res_dict_list.append(
                "FDR-{}_{}_avg-pre-benign-{:.3f}".format(fdr_alpha, algo, output_dict["benign"]["avg_precision"]))
            all_res_dict_list.append(
                "FDR-{}_{}_avg-acc-{:.3f}".format(fdr_alpha, algo, output_dict["common"]["avg_accuracy"]))

            mal_all_res_dict_list_rec.append(
                "FDR-{}_{}_avg-rec-atk-{:.3f}".format(fdr_alpha, algo, output_dict["attack"]["avg_recall"]))

        try:
            os.mkdir(g_fig_dir)
            print("[!] MKDIR : {}".format(g_fig_dir))
        except Exception as e:
            print("[!] MKDIR : {} is already Exist".format(g_fig_dir))

        # NOTICE
        draw_bar_graph(mal_all_res_dict_list, do_eval_online=do_eval_online, metric="accuracy", desc="mal-detect",
                       do_kisti=do_kisti)
        # draw_bar_graph(benign_all_res_dict_list, do_eval_online=do_eval_online, metric="accuracy", desc="benign-detect", do_kisti=do_kisti)
        # draw_bar_graph(all_res_dict_list, do_eval_online=do_eval_online, metric="accuracy", desc="", do_kisti=do_kisti)
        # draw_bar_graph(mal_all_res_dict_list_rec, do_eval_online=do_eval_online, metric="recall", desc="mal-detect", do_kisti=do_kisti)

        if do_eval_online:  # NOTICE: learning curve only for acc
            draw_detailed_online(mal_online_acc_dict_list, do_kisti=do_kisti)
            tmp = calc_real_online(mal_online_acc_dict_list, mal_online_spt_list_dict)
            draw_online_learning_curve(tmp, do_kisti=do_kisti)


def main():
    print("################################################################################")
    print("[!] Experiment for INFOCOM Poster")
    print("[!] Poster Abstract: Encrypted Malware Traffic Detection using Incremental Learning ")
    print("################################################################################")

    # NOTICE NSL-KDD
    # experiment_incremental(bin_clf=is_bin_clf, do_eval_online=True, do_kisti=False)
    # experiment_incremental(bin_clf=is_bin_clf, do_eval_online=False, do_kisti=False)

    # NOTICE KISTI
    experiment_incremental(bin_clf=is_bin_clf, do_eval_online=True, do_kisti=True)
    experiment_incremental(bin_clf=is_bin_clf, do_eval_online=False, do_kisti=True)


if __name__ == "__main__":
    print("HELLO MAIN")
    main()
