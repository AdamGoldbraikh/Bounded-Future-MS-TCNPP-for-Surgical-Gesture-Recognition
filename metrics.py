#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com

import os
import numpy as np
import argparse
from termcolor import colored, cprint
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
import statistics as st

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(
            p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union) * \
            ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def pars_ground_truth(gt_source):
    contant = []
    for line in gt_source:
        info = line.split()
        line_contant = [info[2]] * (int(info[1]) - int(info[0]) + 1)
        contant = contant + line_contant
    return contant


def metric_calculation(ground_truth_path, recognition_list, list_of_videos, suffix="", is_test=False):
    overlap = [.1, .25, .5]
    results_dict = {"Acc "+suffix: None, "Edit "+suffix: None, "F1-macro "+suffix: None,
                    F"F1@{int(overlap[0] * 100) } "+suffix: None, F"F1@{int(overlap[1] * 100) } "+suffix: None,
                    F"F1@{int(overlap[2] * 100)} "+suffix: None
                    }
    results_dict_of_lists = {"Acc "+suffix: None, "Edit "+suffix: None, "F1-macro "+suffix: None,
                             F"F1@{int(overlap[0] * 100) } "+suffix: None, F"F1@{int(overlap[1] * 100) } "+suffix: None,
                             F"F1@{int(overlap[2] * 100)} "+suffix: None
                             }

    gt_list = []
    acc_list = []
    f1_macro_list = []
    edit_list = []
    f1_10_list = []
    f1_25_list = []
    f1_50_list = []

    for i, seq in enumerate(list_of_videos):
        file_ptr = open(os.path.join(ground_truth_path,
                        seq.split('.')[0] + '.txt'), 'r')
        gt_source = file_ptr.read().split('\n')[:-1]
        gt_content = pars_ground_truth(gt_source)

        gt_list.append(gt_content)
        recog_content = recognition_list[i]
        number_of_frames_to_compare = min(len(gt_content), len(recog_content))
        acc_list.append(100*(sum(gt_content[:number_of_frames_to_compare] ==
                        recog_content[:number_of_frames_to_compare]) / number_of_frames_to_compare))
        edit_list.append(edit_score(recog_content, gt_content))
        f1_macro_list.append(100 * f1_score(gt_content[:number_of_frames_to_compare],
                             recog_content[:number_of_frames_to_compare], average='macro'))

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            precision = float(0)
            recall = float(0)
            f1 = float(0)
            if tp1 + fp1 > 0:
                precision = tp1 / float(tp1 + fp1)
            if tp1 + fn1 > 0:
                recall = tp1 / float(tp1 + fn1)
            if precision + recall > 0:
                f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = np.nan_to_num(f1) * 100
            if s == 0:
                f1_10_list.append(f1)
            if s == 1:
                f1_25_list.append(f1)
            if s == 2:
                f1_50_list.append(f1)

    color = "yellow"
    print(colored("Acc: %.4f" % st.mean(acc_list), color))
    results_dict["Acc "+suffix] = st.mean(acc_list)
    print(colored("F1-macro: %.4f" % st.mean(f1_macro_list), color))
    results_dict["F1-macro "+suffix] = st.mean(f1_macro_list)
    print(colored("Edit: %.4f" % st.mean(edit_list), color))
    results_dict["Edit "+suffix] = st.mean(edit_list)
    print(colored('F1@%0.0f: %.4f' %
          (overlap[0]*100, st.mean(f1_10_list)), color))
    results_dict[F"F1@{int(overlap[0] * 100)} " + suffix] = st.mean(f1_10_list)
    print(colored('F1@%0.0f: %.4f' %
          (overlap[1]*100, st.mean(f1_25_list)), color))
    results_dict[F"F1@{int(overlap[1] * 100)} " + suffix] = st.mean(f1_25_list)
    print(colored('F1@%0.0f: %.4f' %
          (overlap[2]*100, st.mean(f1_50_list)), color))
    results_dict[F"F1@{int(overlap[2] * 100)} " + suffix] = st.mean(f1_50_list)
    results_dict_of_lists["Acc "+suffix] = acc_list
    results_dict_of_lists["F1-macro "+suffix] = f1_macro_list
    results_dict_of_lists["Edit "+suffix] = edit_list
    results_dict_of_lists[F"F1@{int(overlap[0] * 100)} " + suffix] = f1_10_list
    results_dict_of_lists[F"F1@{int(overlap[1] * 100)} " + suffix] = f1_25_list
    results_dict_of_lists[F"F1@{int(overlap[2] * 100)} " + suffix] = f1_50_list
    if not is_test:
        return results_dict, gt_list
    if is_test:
        return results_dict_of_lists, gt_list


def metric_calculation_analysis(gt_list, recognition_list, sampling=1, suffix=""):
    overlap = [.1, .25, .5]
    results_dict = {"Acc "+suffix: None, "Edit "+suffix: None, "F1-macro "+suffix: None,
                    F"F1@{int(overlap[0] * 100) } "+suffix: None, F"F1@{int(overlap[1] * 100) } "+suffix: None,
                    F"F1@{int(overlap[2] * 100)} "+suffix: None
                    }
    results_dict_of_lists = {"Acc "+suffix: None, "Edit "+suffix: None, "F1-macro "+suffix: None,
                             F"F1@{int(overlap[0] * 100) } "+suffix: None, F"F1@{int(overlap[1] * 100) } "+suffix: None,
                             F"F1@{int(overlap[2] * 100)} "+suffix: None
                             }

    acc_list = []
    f1_macro_list = []
    edit_list = []
    f1_10_list = []
    f1_25_list = []
    f1_50_list = []

    for i in range(len(recognition_list)):
        gt_content_ = gt_list[i]
        recog_content_ = recognition_list[i]

        for j in range(len(recog_content_)):
            gt_content = gt_content_[j]
            recog_content = recog_content_[j]
            gt_content = gt_content[::sampling]
            recog_content = recog_content[::sampling]

            number_of_frames_to_compare = min(
                len(gt_content), len(recog_content))
            acc_list.append(100*(sum(gt_content[:number_of_frames_to_compare] ==
                            recog_content[:number_of_frames_to_compare]) / number_of_frames_to_compare))
            edit_list.append(edit_score(recog_content, gt_content))
            f1_macro_list.append(
                100 * f1_score(gt_content[:number_of_frames_to_compare], recog_content[:number_of_frames_to_compare], average='macro'))

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
                precision = float(0)
                recall = float(0)
                f1 = float(0)
                if tp1 + fp1 > 0:
                    precision = tp1 / float(tp1 + fp1)
                if tp1 + fn1 > 0:
                    recall = tp1 / float(tp1 + fn1)
                if precision + recall > 0:
                    f1 = 2.0 * (precision * recall) / (precision + recall)
                f1 = np.nan_to_num(f1) * 100
                if s == 0:
                    f1_10_list.append(f1)
                if s == 1:
                    f1_25_list.append(f1)
                if s == 2:
                    f1_50_list.append(f1)

    color = "yellow"
    print(colored("Acc: %.4f" % st.mean(acc_list), color))
    results_dict["Acc "+suffix] = st.mean(acc_list)
    print(colored("F1-macro: %.4f" % st.mean(f1_macro_list), color))
    results_dict["F1-macro "+suffix] = st.mean(f1_macro_list)
    print(colored("Edit: %.4f" % st.mean(edit_list), color))
    results_dict["Edit "+suffix] = st.mean(edit_list)
    print(colored('F1@%0.0f: %.4f' %
          (overlap[0]*100, st.mean(f1_10_list)), color))
    results_dict[F"F1@{int(overlap[0] * 100)} " + suffix] = st.mean(f1_10_list)
    print(colored('F1@%0.0f: %.4f' %
          (overlap[1]*100, st.mean(f1_25_list)), color))
    results_dict[F"F1@{int(overlap[1] * 100)} " + suffix] = st.mean(f1_25_list)
    print(colored('F1@%0.0f: %.4f' %
          (overlap[2]*100, st.mean(f1_50_list)), color))
    results_dict[F"F1@{int(overlap[2] * 100)} " + suffix] = st.mean(f1_50_list)
    results_dict_of_lists["Acc "+suffix] = acc_list
    results_dict_of_lists["F1-macro "+suffix] = f1_macro_list
    results_dict_of_lists["Edit "+suffix] = edit_list
    results_dict_of_lists[F"F1@{int(overlap[0] * 100)} " + suffix] = f1_10_list
    results_dict_of_lists[F"F1@{int(overlap[1] * 100)} " + suffix] = f1_25_list
    results_dict_of_lists[F"F1@{int(overlap[2] * 100)} " + suffix] = f1_50_list
    return results_dict_of_lists


def metric_calculation_backup(ground_truth_path, recognition_list, list_of_videos, suffix=""):
    overlap = [.1, .25, .5]
    results_dict = {"Acc "+suffix: None, "Edit "+suffix: None, "F1-macro "+suffix: None,
                    F"F1@{int(overlap[0] * 100) } "+suffix: None, F"F1@{int(overlap[1] * 100) } "+suffix: None,
                    F"F1@{int(overlap[2] * 100)} "+suffix: None
                    }
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0
    gt_list = []
    all_gt = []
    all_recogs = []
    for i, seq in enumerate(list_of_videos):

        file_ptr = open(ground_truth_path + seq.split('.')[0] + '.txt', 'r')
        gt_source = file_ptr.read().split('\n')[:-1]
        gt_content = pars_ground_truth(gt_source)

        gt_list.append(gt_content)
        recog_content = recognition_list[i]
        all_gt = all_gt + gt_content[:min(len(gt_content), len(recog_content))]
        all_recogs = all_recogs + \
            list(recog_content[:min(len(gt_content), len(recog_content))])

        for i in range(min(len(gt_content), len(recog_content))):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    color = "yellow"
    print(colored("Acc: %.4f" % (100*float(correct)/total), color))
    f1_macro = f1_score(all_gt, all_recogs, average='macro')
    results_dict["F1-macro "+suffix] = f1_macro
    labels = list(set(all_gt+all_recogs))
    f1_per_class = f1_score(all_gt, all_recogs, average='macro', labels=labels)

    print(colored("F1-macro: %.4f" % (100 * float(f1_macro)), color))
    print(colored('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)), color))
    acc = (100*float(correct)/total)
    results_dict["Acc "+suffix] = acc
    edit = ((1.0*edit)/len(list_of_videos))
    results_dict["Edit "+suffix] = edit
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])

        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        results_dict[F"F1@{int(overlap[s] * 100)} " + suffix] = f1
        print(colored('F1@%0.2f: %.4f' % (overlap[s], f1), color))

    return results_dict, gt_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')

    args = parser.parse_args()

    ground_truth_path = "./data/"+args.dataset+"/groundTruth/"
    recog_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    file_list = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]

        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    print("Acc: %.4f" % (100*float(correct)/total))
    print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    acc = (100*float(correct)/total)
    edit = ((1.0*edit)/len(list_of_videos))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])

        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))


if __name__ == '__main__':
    main()
