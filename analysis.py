# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
from metrics import *
from batch_gen import BatchGenerator, BatchGenerator_1Modal
from Trainer import Trainer, Trainer_1Modal
from visualization import plot_seq_gestures, save_all_seq
import pandas as pd
import matplotlib.font_manager
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime
from metrics import pars_ground_truth
import torch
import os
import argparse
import random
import numpy as np


def get_gt(ground_truth_path, list_of_videos):
    gt_list = []
    for i, seq in enumerate(list_of_videos):

        file_ptr = open(ground_truth_path + seq.split('.')[0] + '.txt', 'r')
        gt_source = file_ptr.read().split('\n')[:-1]
        gt_content = pars_ground_truth(gt_source)

        gt_list.append(gt_content)
    return gt_list


def predict(trainer, model_dir, features_path, list_of_vids, actions_dict_gestures, actions_dict_tools, device, sample_rate, task, network):
    trainer.model.eval()
    with torch.no_grad():
        trainer.model.to(device)
        trainer.model.load_state_dict(torch.load(
            model_dir + "/" + network + "_" + task + ".model"))
        recognition1_list = []
        recognition2_list = []
        recognition3_list = []
        for seq in list_of_vids:

            features = np.load(features_path+"/" + seq.split('.')[0] + '.npy')
            if batch_gen.normalization == "Min-max":
                numerator = features.T - batch_gen.min
                denominator = batch_gen.max - batch_gen.min
                features = (numerator / denominator).T
            elif batch_gen.normalization == "Standard":
                numerator = features.T - batch_gen.mean
                denominator = batch_gen.std
                features = (numerator / denominator).T
            elif batch_gen.normalization == "samplewise_SD":
                samplewise_meam = features.mean(axis=1)
                samplewise_std = features.std(axis=1)
                numerator = features.T - samplewise_meam
                denominator = samplewise_std
                features = (numerator / denominator).T

            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if task == "multi-taks":
                if network == "LSTM" or network == "GRU":
                    predictions1, predictions2, predictions3 = trainer.model(
                        input_x, torch.tensor([features.shape[1]]))
                    predictions1 = predictions1.unsqueeze_(0)
                    predictions1 = torch.nn.Softmax(dim=2)(predictions1)

                    predictions2 = predictions2.unsqueeze_(0)
                    predictions2 = torch.nn.Softmax(dim=2)(predictions2)
                    predictions3 = predictions3.unsqueeze_(0)
                    predictions3 = torch.nn.Softmax(dim=2)(predictions3)
                else:
                    predictions1, predictions2, predictions3 = trainer.model(
                        input_x)
            elif task == "tools":
                if network == "LSTM" or network == "GRU":
                    predictions2, predictions3 = trainer.model(
                        input_x, torch.tensor([features.shape[1]]))
                    predictions2 = predictions2.unsqueeze_(0)
                    predictions2 = torch.nn.Softmax(dim=2)(predictions2)
                    predictions3 = predictions3.unsqueeze_(0)
                    predictions3 = torch.nn.Softmax(dim=2)(predictions3)

                else:
                    predictions2, predictions3 = trainer.model(input_x)
            else:
                if network == "LSTM" or network == "GRU":
                    predictions1 = trainer.model(
                        input_x, torch.tensor([features.shape[1]]))
                    predictions1 = predictions1[0].unsqueeze_(0)
                    predictions1 = torch.nn.Softmax(dim=2)(predictions1)
                else:
                    predictions1 = trainer.model(
                        input_x, torch.tensor([features.shape[1]]))[0]

            if task == "multi-taks" or task == "gestures":
                _, predicted1 = torch.max(predictions1[-1].data, 1)
                predicted1 = predicted1.squeeze()

            if task == "multi-taks" or task == "tools":
                _, predicted2 = torch.max(predictions2[-1].data, 1)
                _, predicted3 = torch.max(predictions3[-1].data, 1)
                predicted2 = predicted2.squeeze()
                predicted3 = predicted3.squeeze()

            recognition1 = []
            recognition2 = []
            recognition3 = []
            if task == "multi-taks" or task == "gestures":
                for i in range(len(predicted1)):
                    recognition1 = np.concatenate((recognition1, [list(actions_dict_gestures.keys())[
                        list(actions_dict_gestures.values()).index(
                            predicted1[i].item())]] * sample_rate))
                recognition1_list.append(recognition1)
            if task == "multi-taks" or task == "tools":

                for i in range(len(predicted2)):
                    recognition2 = np.concatenate((recognition2, [list(actions_dict_tools.keys())[
                        list(actions_dict_tools.values()).index(
                            predicted2[i].item())]] * sample_rate))
                recognition2_list.append(recognition2)

                for i in range(len(predicted3)):
                    recognition3 = np.concatenate((recognition3, [list(actions_dict_tools.keys())[
                        list(actions_dict_tools.values()).index(
                            predicted3[i].item())]] * sample_rate))
                recognition3_list.append(recognition3)

        return recognition1_list, recognition2_list, recognition3_list


def actions_list_to_ids(recognition_list, actions_dict):
    """

    :param recognition_list: list os lists of labels
    :param actions_dict: dicts from labels to action ids
    :return: list of lists of ids
    """
    output = []
    for video_labels in recognition_list:
        list_of_ids = []
        for label in list(video_labels):
            list_of_ids.append(actions_dict[label])
        output.append(list_of_ids)
    return output


def prepare_for_visual_sammary(recognition_id_list, gt_id_list, list_of_vidios):
    merged_ids_list = []
    name_list = []
    for recog_list, gt_list, video_name in zip(recognition_id_list, gt_id_list, list_of_vidios):
        merged_ids_list.append(recog_list)
        name_list.append(video_name[:-4] + " predicted")
        merged_ids_list.append(gt_list)
        name_list.append(video_name[:-4] + " ground truth")
    return merged_ids_list, name_list


def conf_mat_calc(all_recogs, all_gts, labels):
    flatten_recogs = []
    flatten_gt = []
    for split in all_gts:
        for seq in split:
            flatten_gt += seq

    for split in all_recogs:
        for seq in split:
            flatten_recogs += seq.tolist()

    distribution = confusion_matrix(flatten_gt, flatten_recogs, labels=labels)
    # print(distribution)
    distribution = (distribution.transpose() /
                    np.sum(distribution, 1)).transpose()
    # distribution = (distribution.transpose() / np.sum(distribution)).transpose()

    conf_mat_draw(distribution, labels)


def conf_mat_draw(distribution, labels):
    if "G0" in labels:
        ax = sns.heatmap(distribution, annot=True,
                         xticklabels=['no gesture', "needle passing", "pull the suture", "instrument tie", "lay the knot",
                                      "cut the suture"],
                         yticklabels=['no gesture', "needle passing", "pull the suture", "instrument tie", "lay the knot",
                                      "cut the suture"], fmt='.3f', cmap=sns.color_palette("mako"))
        plt.xticks(rotation=45)

    else:
        ax = sns.heatmap(distribution, annot=True,
                         xticklabels=["no tool", "needle driver", "forceps",
                                      "scissors"],
                         yticklabels=["no tool", "needle driver", "forceps",
                                      "scissors"], fmt='.3f', cmap=sns.color_palette("mako"))

    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    plt.show()


def findToolConditiondOnGesture(all_recogs_tools, all_gts_tools, all_gts_gestures):
    gt_pairs_total = 0
    correct_pairs = 0
    total = 0
    for split_tools, split_gts_tools, split_gts_gestures in zip(all_recogs_tools, all_gts_tools, all_gts_gestures):
        for recog_tool, gt_tool, gt_gestures in zip(split_tools, split_gts_tools, split_gts_gestures):
            min_len = min(len(recog_tool), len(gt_tool), len(gt_gestures))
            recog_tool = recog_tool[:min_len]
            gt_tool = gt_tool[:min_len]
            gt_gestures = gt_gestures[:min_len]
            for index, (gesture, tool) in enumerate(zip(gt_gestures, gt_tool)):
                total += 1
                if gesture == 'G5' and tool == "T1":
                    gt_pairs_total += 1
                    if tool == recog_tool[index]:
                        correct_pairs += 1
    print("% of success among the relevant frames")
    print(100*(correct_pairs)/gt_pairs_total)
    print("% relevant frames")
    print(100*(gt_pairs_total)/total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['VTS'], default="VTS")
    parser.add_argument(
        '--task', choices=['gestures', 'tools', 'multi-taks'], default="gestures")
    parser.add_argument('--network',
                        choices=['MS-TCN2', 'MS-TCN2_ISR', 'LSTM', 'GRU', 'MS-LSTM-TCN', 'MS-TCN-LSTM', 'MS-GRU-TCN',
                                 'MS-TCN-GRU'], default="MS-TCN2")
    parser.add_argument(
        '--split', choices=['0', '1', '2', '3', '4', 'all'], default='all')
    parser.add_argument('--features_dim', default=1280, type=int)
    parser.add_argument('--lr', default='0', type=float)
    parser.add_argument('--num_epochs', default=0, type=int)

    # Architectuyre
    parser.add_argument('--eval_rate', default=0, type=int)
    parser.add_argument('--num_f_maps', default=64, type=int)

    parser.add_argument('--num_layers_TCN', default=11, type=int)
    parser.add_argument('--normalization', choices=['Min-max', 'Standard', 'samplewise_SD', 'none'], default='none',
                        type=str)
    parser.add_argument('--num_R', default=3, type=int)

    parser.add_argument('--hidden_dim_rnn', default=128, type=int)
    parser.add_argument('--num_layers_rnn', default=2, type=int)
    parser.add_argument('--sample_rate', default=1, type=int)
    parser.add_argument('--secondary_sampling', default=3, type=int)

    parser.add_argument('--loss_tau', default=16, type=float)
    parser.add_argument('--loss_lambda', default=0, type=float)
    parser.add_argument('--dropout_TCN', default=0, type=float)
    parser.add_argument('--dropout_RNN', default=0, type=float)
    parser.add_argument('--offline_mode', default=True, type=bool)
    parser.add_argument(
        '--project', default="Offline RNN nets Sensor paper Final", type=str)
    parser.add_argument('--use_gpu_num', default="1", type=str)
    parser.add_argument('--upload', default=False, type=bool)
    parser.add_argument('--filtered_data', default=True, type=bool)
    parser.add_argument('--debagging', default=False, type=bool)
    parser.add_argument('--hyper_parameter_tuning', default=False, type=bool)
    parser.add_argument('--azure', default=False, type=bool)
    eval_sampling = 1
    args = parser.parse_args()
    args.specific_seq = ""

    print(args)
    seed = 1642708740
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bz = 1
    offline_mode = True

    # use the full temporal resolution @ 30Hz

    list_of_splits = []
    if len(args.split) == 1:
        list_of_splits.append(int(args.split))

    elif args.dataset == "VTS":
        list_of_splits = list(range(0, 5))
    else:
        list_of_splits = list(range(0, 8))

    if args.network == "MS-TCN-LSTM":

        list_of_experiments = ["24.05.2022 10:03:41  task:gestures splits: all net: MS-TCN-LSTM is Offline: True",
                               "24.05.2022 10:04:23  task:gestures splits: all net: MS-TCN-LSTM is Offline: True",
                               "24.05.2022 10:04:34  task:gestures splits: all net: MS-TCN-LSTM is Offline: True",
                               "24.05.2022 16:37:38  task:gestures splits: all net: MS-TCN-LSTM is Offline: True",
                               "24.05.2022 16:37:50  task:gestures splits: all net: MS-TCN-LSTM is Offline: True",
                               "24.05.2022 16:38:43  task:gestures splits: all net: MS-TCN-LSTM is Offline: True",
                               "24.05.2022 16:41:18  task:gestures splits: all net: MS-TCN-LSTM is Offline: True",
                               "24.05.2022 16:41:28  task:gestures splits: all net: MS-TCN-LSTM is Offline: True"]

    elif args.network == "GRU":
        hidden_dim_rnn = 128
        num_layers_rnn = 3
        sample_rate = 4
        if args.task == "multi-taks":
            list_of_experiments = ["08.02.2022 15:11:41  task:multi-taks splits: all net: GRU is Offline: True",
                                   "08.02.2022 15:10:13  task:multi-taks splits: all net: GRU is Offline: True",
                                   "08.02.2022 15:08:52  task:multi-taks splits: all net: GRU is Offline: True",
                                   "08.02.2022 15:07:30  task:multi-taks splits: all net: GRU is Offline: True",
                                   "08.02.2022 15:05:02  task:multi-taks splits: all net: GRU is Offline: True",
                                   "08.02.2022 15:03:32  task:multi-taks splits: all net: GRU is Offline: True",
                                   "08.02.2022 15:02:02  task:multi-taks splits: all net: GRU is Offline: True",
                                   "08.02.2022 15:00:36  task:multi-taks splits: all net: GRU is Offline: True"]

        elif args.task == "tools":
            list_of_experiments = ["08.02.2022 14:00:25  task:tools splits: all net: GRU is Offline: True",
                                   "08.02.2022 14:01:56  task:tools splits: all net: GRU is Offline: True",
                                   "08.02.2022 14:03:28  task:tools splits: all net: GRU is Offline: True",
                                   "08.02.2022 14:04:57  task:tools splits: all net: GRU is Offline: True",
                                   "08.02.2022 14:06:40  task:tools splits: all net: GRU is Offline: True",
                                   "08.02.2022 14:09:10  task:tools splits: all net: GRU is Offline: True",
                                   "08.02.2022 14:11:09  task:tools splits: all net: GRU is Offline: True",
                                   "08.02.2022 14:12:49  task:tools splits: all net: GRU is Offline: True"]
        elif args.task == "gestures":
            # list_of_experiments = ["24.01.2022 20:57:14  task:gestures splits: all net: GRU is Offline: True,"
            # "24.01.2022 20:52:25  task:gestures splits: all net: GRU is Offline: True"]

            list_of_experiments = ["24.01.2022 20:50:12  task:gestures splits: all net: GRU is Offline: True",
                                   "24.01.2022 20:52:25  task:gestures splits: all net: GRU is Offline: True",
                                   "24.01.2022 20:51:21  task:gestures splits: all net: GRU is Offline: True",
                                   "24.01.2022 20:53:16  task:gestures splits: all net: GRU is Offline: True",
                                   "24.01.2022 20:54:08  task:gestures splits: all net: GRU is Offline: True",
                                   "24.01.2022 20:55:07  task:gestures splits: all net: GRU is Offline: True",
                                   "24.01.2022 20:56:00  task:gestures splits: all net: GRU is Offline: True",
                                   "24.01.2022 20:57:14  task:gestures splits: all net: GRU is Offline: True"]

    elif args.network == "MS-TCN2":
        num_layers_PG = 13
        num_layers_R = 13
        num_R = 1
        num_f_maps = 128
        sample_rate = 1

        if args.task == "multi-taks":
            list_of_experiments = ["07.02.2022 19:31:12  task:multi-taks splits: all net: MS-TCN2 is Offline: True",
                                   "07.02.2022 19:33:32  task:multi-taks splits: all net: MS-TCN2 is Offline: True",
                                   "07.02.2022 19:36:06  task:multi-taks splits: all net: MS-TCN2 is Offline: True",
                                   "07.02.2022 19:37:16  task:multi-taks splits: all net: MS-TCN2 is Offline: True",
                                   "07.02.2022 19:38:31  task:multi-taks splits: all net: MS-TCN2 is Offline: True",
                                   "07.02.2022 19:39:29  task:multi-taks splits: all net: MS-TCN2 is Offline: True",
                                   "07.02.2022 19:40:37  task:multi-taks splits: all net: MS-TCN2 is Offline: True",
                                   "07.02.2022 19:41:45  task:multi-taks splits: all net: MS-TCN2 is Offline: True"]
        elif args.task == "tools":
            list_of_experiments = ["08.02.2022 09:11:38  task:tools splits: all net: MS-TCN2 is Offline: True",
                                   "08.02.2022 09:13:21  task:tools splits: all net: MS-TCN2 is Offline: True",
                                   "08.02.2022 09:14:34  task:tools splits: all net: MS-TCN2 is Offline: True",
                                   "08.02.2022 09:15:56  task:tools splits: all net: MS-TCN2 is Offline: True",
                                   "08.02.2022 09:18:00  task:tools splits: all net: MS-TCN2 is Offline: True",
                                   "08.02.2022 09:21:46  task:tools splits: all net: MS-TCN2 is Offline: True",
                                   "08.02.2022 09:23:06  task:tools splits: all net: MS-TCN2 is Offline: True",
                                   "08.02.2022 09:24:01  task:tools splits: all net: MS-TCN2 is Offline: True"]
        elif args.task == "gestures":
            list_of_experiments = [
                "14.06.2022 14:33:09  task:gestures splits: all net: MS-TCN2 is Offline: True"]

            # list_of_experiments = ["20.01.2022 21:59:00  task:gestures splits: all net: MS-TCN2 is Offline: True",
            #                        "20.01.2022 22:01:37  task:gestures splits: all net: MS-TCN2 is Offline: True",
            #                        "20.01.2022 22:03:33  task:gestures splits: all net: MS-TCN2 is Offline: True",
            #                        "20.01.2022 22:04:57  task:gestures splits: all net: MS-TCN2 is Offline: True",
            #                        "20.01.2022 22:06:03  task:gestures splits: all net: MS-TCN2 is Offline: True",
            #                        "20.01.2022 22:07:23  task:gestures splits: all net: MS-TCN2 is Offline: True",
            #                        "20.01.2022 22:08:48  task:gestures splits: all net: MS-TCN2 is Offline: True",
            #                        "20.01.2022 22:10:12  task:gestures splits: all net: MS-TCN2 is Offline: True"]

    elif args.network == "LSTM":
        hidden_dim_rnn = 256
        num_layers_rnn = 3
        sample_rate = 5
        if args.task == "multi-taks":

            list_of_experiments = ["08.02.2022 16:36:44  task:multi-taks splits: all net: LSTM is Offline: True",
                                   "08.02.2022 16:38:11  task:multi-taks splits: all net: LSTM is Offline: True",
                                   "08.02.2022 16:40:41  task:multi-taks splits: all net: LSTM is Offline: True",
                                   "08.02.2022 16:41:56  task:multi-taks splits: all net: LSTM is Offline: True",
                                   "08.02.2022 16:43:15  task:multi-taks splits: all net: LSTM is Offline: True",
                                   "08.02.2022 16:45:04  task:multi-taks splits: all net: LSTM is Offline: True",
                                   "08.02.2022 16:46:38  task:multi-taks splits: all net: LSTM is Offline: True",
                                   "08.02.2022 20:39:57  task:multi-taks splits: all net: LSTM is Offline: True"]
        elif args.task == "tools":
            list_of_experiments = ["08.02.2022 22:10:08  task:tools splits: all net: LSTM is Offline: True",
                                   "08.02.2022 22:11:45  task:tools splits: all net: LSTM is Offline: True",
                                   "08.02.2022 22:13:21  task:tools splits: all net: LSTM is Offline: True",
                                   "08.02.2022 22:16:28  task:tools splits: all net: LSTM is Offline: True",
                                   "08.02.2022 22:17:46  task:tools splits: all net: LSTM is Offline: True",
                                   "08.02.2022 22:25:14  task:tools splits: all net: LSTM is Offline: True",
                                   "08.02.2022 22:26:59  task:tools splits: all net: LSTM is Offline: True",
                                   "08.02.2022 22:30:01  task:tools splits: all net: LSTM is Offline: True"]
        elif args.task == "gestures":

            list_of_experiments = ["23.01.2022 13:29:18  task:gestures splits: all net: LSTM is Offline: True",
                                   "23.01.2022 13:29:16  task:gestures splits: all net: LSTM is Offline: True",
                                   "23.01.2022 13:29:03  task:gestures splits: all net: LSTM is Offline: True",
                                   "23.01.2022 13:28:31  task:gestures splits: all net: LSTM is Offline: True",
                                   "23.01.2022 13:28:29  task:gestures splits: all net: LSTM is Offline: True",
                                   "23.01.2022 13:28:25  task:gestures splits: all net: LSTM is Offline: True",
                                   "23.01.2022 13:28:10  task:gestures splits: all net: LSTM is Offline: True",
                                   "23.01.2022 13:25:15  task:gestures splits: all net: LSTM is Offline: True"]

    else:
        raise NotImplementedError

    all_recogs1 = []
    all_recogs2 = []
    all_recogs3 = []
    all_gt1 = []
    all_gt2 = []
    all_gt3 = []
    list_of_names = []
    seed_id = []
    split_id = []
    all_names = []

    for k, experiment_name in enumerate(list_of_experiments):

        summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name

        for split_num in list_of_splits:

            print("split number: " + str(split_num))
            args.split = str(split_num)

            folds_folder = "./data/" + args.dataset + "/folds"

            features_path = os.path.join(
                "data", args.dataset, "features", "fold " + str(split_num))

            gt_path_gestures = "./data/" + args.dataset + "/transcriptions_gestures/"
            gt_path_tools_left = "./data/" + args.dataset + "/transcriptions_tools_left/"
            gt_path_tools_right = "./data/" + args.dataset + "/transcriptions_tools_right/"

            mapping_gestures_file = "./data/" + args.dataset + "/mapping_gestures.txt"
            mapping_tool_file = "./data/" + args.dataset + "/mapping_tools.txt"

            model_dir = "./models/" + args.dataset + "/" + \
                experiment_name + "/split" + args.split

            file_ptr = open(mapping_gestures_file, 'r')
            actions = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            actions_dict_gestures = dict()
            for a in actions:
                actions_dict_gestures[a.split()[1]] = int(a.split()[0])
            num_classes_tools = 0
            actions_dict_tools = dict()
            if args.dataset == "VTS":
                file_ptr = open(mapping_tool_file, 'r')
                actions = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                for a in actions:
                    actions_dict_tools[a.split()[1]] = int(a.split()[0])
                num_classes_tools = len(actions_dict_tools)

            num_classes_gestures = len(actions_dict_gestures)

            if args.task == "gestures":
                num_classes_list = [num_classes_gestures]
            elif args.dataset == "VTS" and args.task == "tools":
                num_classes_list = [num_classes_tools, num_classes_tools]
            elif args.dataset == "VTS" and args.task == "multi-taks":
                num_classes_list = [num_classes_gestures,
                                    num_classes_tools, num_classes_tools]

            trainer = Trainer_1Modal(args.num_layers_TCN, args.num_layers_TCN, args.num_R, args.num_f_maps, args.features_dim, num_classes_list,
                                     offline_mode=offline_mode,
                                     tau=0, lambd=0, hidden_dim_rnn=args.hidden_dim_rnn,
                                     num_layers_rnn=args.num_layers_rnn,
                                     dropout_TCN=args.dropout_TCN, dropout_RNN=args.dropout_RNN, task=args.task, device=device,
                                     network=args.network,
                                     secondary_sampling=args.secondary_sampling,
                                     hyper_parameter_tuning=False, debagging=False)

            batch_gen = BatchGenerator_1Modal(num_classes_gestures, num_classes_tools, actions_dict_gestures, actions_dict_tools,
                                              features_path, split_num, folds_folder, gt_path_gestures, gt_path_tools_left,
                                              gt_path_tools_right, sample_rate=args.sample_rate, normalization=args.normalization,
                                              task=args.task)

            eval_dict = {"features_path": features_path, "actions_dict_gestures": actions_dict_gestures,
                         "actions_dict_tools": actions_dict_tools, "device": device, "sample_rate": args.sample_rate,
                         "eval_rate": 1,
                         "gt_path_gestures": gt_path_gestures, "gt_path_tools_left": gt_path_tools_left,
                         "gt_path_tools_right": gt_path_tools_right, "task": args.task}

            list_of_vids = batch_gen.list_of_test_examples
            list_of_names += list_of_vids
            seed_id += [k] * len(list_of_vids)
            split_id += [split_num] * len(list_of_vids)

            if args.specific_seq != "":
                if args.specific_seq+".csv" not in list_of_vids:
                    continue
                else:
                    list_of_vids = [args.specific_seq+".csv"]

            recognition1_list, recognition2_list, recognition3_list = predict(trainer, model_dir, features_path, list_of_vids, actions_dict_gestures, actions_dict_tools, device,
                                                                              args.sample_rate, args.task, args.network)

            if args.task == "multi-taks" or args.task == "gestures":
                print("gestures results")
                gt_list_1 = get_gt(ground_truth_path=gt_path_gestures,
                                   list_of_videos=list_of_vids)
                for i in range(len(gt_list_1)):
                    min_len = min(len(gt_list_1[i]), len(recognition1_list[i]))
                    gt_list_1[i] = gt_list_1[i][:min_len]
                    recognition1_list[i] = recognition1_list[i][:min_len]

                all_names.append(list_of_vids)
                all_recogs1.append(recognition1_list)
                all_gt1.append(gt_list_1)

            if args.task == "multi-taks" or args.task == "tools":
                gt_list_2 = get_gt(
                    ground_truth_path=gt_path_tools_right, list_of_videos=list_of_vids)

                for i in range(len(gt_list_2)):
                    min_len = min(len(gt_list_2[i]), len(recognition2_list[i]))
                    gt_list_2[i] = gt_list_2[i][:min_len]
                    recognition2_list[i] = recognition2_list[i][:min_len]

                gt_list_3 = get_gt(ground_truth_path=gt_path_tools_left,
                                   list_of_videos=list_of_vids)

                for i in range(len(gt_list_3)):
                    min_len = min(len(gt_list_3[i]), len(recognition3_list[i]))
                    gt_list_3[i] = gt_list_3[i][:min_len]
                    recognition3_list[i] = recognition3_list[i][:min_len]

                all_recogs2.append(recognition2_list)
                all_recogs3.append(recognition3_list)
                all_gt2.append(gt_list_2)
                all_gt3.append(gt_list_3)
    metadata = {"procedure name": list_of_names,
                "split": split_id, "seed id": seed_id}

    # if args.task == "multi-taks":
    #     findToolConditiondOnGesture(all_recogs3, all_gt3, all_gt1)
    if args.task == "multi-taks" or args.task == "gestures":
        results_gesture = metric_calculation_analysis(
            all_gt1, all_recogs1, eval_sampling)
        metadata.update(results_gesture)
        for i in range(len(all_gt1)):
            for j in range(len(all_gt1[i])):
                all_gt1[i][j] = all_gt1[i][j][::eval_sampling]
                all_recogs1[i][j] = all_recogs1[i][j][::eval_sampling]

        # plot_seq_gestures(all_recogs1, all_gt1,all_names, "gesture", True)
        # plot_seq_gestures(all_recogs1, all_gt1,all_names, "gesture", False)

        save_all_seq(all_recogs1, all_gt1, all_names, "gesture")

        conf_mat_calc(all_recogs1, all_gt1, [
                      "G0", "G1", "G2", "G3", "G4", "G5"])

    if args.task == "multi-taks" or args.task == "tools":
        results_right = metric_calculation_analysis(
            all_gt2, all_recogs2, eval_sampling, suffix=" right")
        results_left = metric_calculation_analysis(
            all_gt3, all_recogs3, eval_sampling, suffix=" left")
        metadata.update(results_right)
        metadata.update(results_left)
        for i in range(len(all_gt2)):
            for j in range(len(all_gt2[i])):
                all_gt2[i][j] = all_gt2[i][j][::eval_sampling]
                all_recogs2[i][j] = all_recogs2[i][j][::eval_sampling]

        for i in range(len(all_gt3)):
            for j in range(len(all_gt3[i])):
                all_gt3[i][j] = all_gt3[i][j][::eval_sampling]
                all_recogs3[i][j] = all_recogs3[i][j][::eval_sampling]

        plot_seq_gestures(all_recogs2, all_gt2, "right hand", True)
        plot_seq_gestures(all_recogs2, all_gt2, "right hand", False)
        plot_seq_gestures(all_recogs3, all_gt3, "left hand", True)
        plot_seq_gestures(all_recogs3, all_gt3, "left hand", False)
        conf_mat_calc(all_recogs2, all_gt2, ["T0", "T1", "T2", "T3"])
        conf_mat_calc(all_recogs3, all_gt3, ["T0", "T1", "T2", "T3"])
    df = pd.DataFrame(metadata)
    df.to_csv(args.network + " " + args.task + ".csv")
