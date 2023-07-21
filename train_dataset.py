from typing import Tuple
import torch
import torch.utils.data as data
import torchvision
from transforms import Stack, ToTorchFormatTensor

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import random
import cv2
from skimage.util import random_noise
import pandas as pd
import math

# import parser


class SequentialTestGestureDataSet(data.Dataset):

    def __init__(self, root_path, video_id, frame_count, transcriptions_dir, gesture_ids,
                 snippet_length=16, sampling_step=6,
                 image_tmpl='img_{:05d}.jpg', video_suffix="_capture2",
                 return_3D_tensor=True, return_dense_labels=True,
                 transform=None, normalize=None):

        self.root_path = root_path
        self.video_name = video_id
        self.video_id = video_id
        self.transcriptions_dir = transcriptions_dir
        self.gesture_ids = gesture_ids
        self.snippet_length = snippet_length
        self.sampling_step = sampling_step
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.return_3D_tensor = return_3D_tensor
        self.return_dense_labels = return_dense_labels
        self.transform = transform
        self.normalize = normalize
        self.frame_count = int(frame_count)

        self.gesture_sequence_per_video = {}
        self.image_data = {}
        self.frame_num_data = {}
        self.labels_data = {}
        self._parse_list_files(video_id)

    def _parse_list_files(self, video_name):
        # depands only on csv files of splits directory
        video_id = video_name
        _frame_count = self.frame_count

        gestures_file = os.path.join(
            self.transcriptions_dir, video_id + ".txt")
        gestures = [[int(x.strip().split(' ')[0]), int(x.strip().split(' ')[1]), x.strip().split(' ')[2]]
                    for x in open(gestures_file)]
        # [start_frame, end_frame, gesture_id]

        _initial_labeled_frame = gestures[0][0]
        _final_labaled_frame = gestures[-1][1]

        _last_rgb_frame = os.path.join(self.root_path, video_id + self.video_suffix,
                                       'img_{:05d}.jpg'.format(_frame_count))

        self.frame_num_data[video_id] = list(
            range(_initial_labeled_frame, _final_labaled_frame, self.sampling_step))
        self._generate_labels_list(video_id, gestures)
        self._preload_images(video_id)
        assert len(self.image_data[video_id]) == len(
            self.labels_data[video_id])

    def _generate_labels_list(self, video_id, gestures):
        labels_list = []

        for frame_num in self.frame_num_data[self.video_name]:
            for gesture in gestures:
                if frame_num >= gesture[0] and frame_num <= gesture[1]:
                    labels_list.append(self.gesture_ids.index(gesture[2]))
                    break
        # for gesture in gestures:
        #     for idx in range(gesture[0],gesture[1]+1):
        #         labels_list.append(self.gesture_ids.index(gesture[2]))
        self.labels_data[video_id] = labels_list

    def _preload_images(self, video_id):
        print("Preloading images from video {}...".format(video_id))
        images = []
        img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        for idx in self.frame_num_data[self.video_name]:
            imgs = self._load_image(img_dir, idx)
            images.extend(imgs)
        self.image_data[video_id] = images

    def _load_image(self, directory, idx):
        img = Image.open(os.path.join(
            directory, self.image_tmpl.format(idx))).convert('RGB')
        return [img]

    def __getitem__(self, index):
        last_index = index + self.snippet_length
        frame_list = list(range(index, last_index))
        target = self._get_snippet_labels(self.video_name, frame_list)
        target = target[-1]
        data = self._get_snippet(self.video_name, frame_list)

        return data, target

    def _get_snippet_labels(self, video_id, frame_list):
        assert self.return_dense_labels
        labels = self.labels_data[video_id]
        target = []
        idx = frame_list[-1]
        target.append(int(labels[idx]))
        return torch.tensor(target, dtype=torch.long)

    def _get_snippet(self, video_id, frame_list):
        snippet = list()
        for idx in frame_list:
            _idx = max(idx, 0)  # padding if required
            img = self.image_data[video_id][_idx]
            snippet.append(img)
        # snippet = rotate_snippet(snippet,0.5)
        # Add_Gaussian_Noise_to_snippet(snippet)
        snippet = self.transform(snippet)
        snippet = [torchvision.transforms.ToTensor()(img) for img in snippet]
        snippet = torch.stack(snippet, 0)
        snippet = self.normalize(snippet)
        snippet = snippet.view(((self.snippet_length,) + snippet.size()[-3:]))
        snippet = snippet.permute(1, 0, 2, 3)
        data = snippet
        return data

    def __len__(self):
        return (len(self.image_data[self.video_name])) - (self.snippet_length - 1)


class GestureTrainSet(data.Dataset):
    def __init__(self, root_path, list_of_list_files, transcriptions_dir, gesture_ids,
                 snippet_length=16, sampling_policies=[[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]], temporal_augmentaion_factor=0.2,
                 image_tmpl='img_{:05d}.jpg', video_suffix="_capture2",
                 return_3D_tensor=True, return_dense_labels=True,
                 transform=None, normalize=None, number_of_samples_per_class=400, debag=False):
        self.debag = debag
        self.root_path = root_path
        self.list_of_list_files = list_of_list_files
        self.transcriptions_dir = transcriptions_dir
        self.gesture_ids = gesture_ids
        self.snippet_length = snippet_length
        self.sampling_policies_list = sampling_policies
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.return_3D_tensor = return_3D_tensor
        self.return_dense_labels = return_dense_labels
        self.transform = transform
        self.normalize = normalize

        self.gesture_sequence_per_video = {}
        self.image_data = {}
        self.frame_num_data = {}
        self.labels_data = {}
        self._parse_list_files(list_of_list_files)
        self.ballanced_data_set = []

        for policy_index, policy in enumerate(self.sampling_policies_list):
            # number_of_frames_in_smallest_class = self._get_the_min_numb_of_frames_per_calas(policy)
            if policy_index == 0:

                num_of_frames_to_choose = number_of_samples_per_class
            else:
                num_of_frames_to_choose = math.floor(
                    number_of_samples_per_class * temporal_augmentaion_factor)
            gesture_dict_per_policy = self._sort_frames_by_gesture(
                policy_index)
            self._random_balancing(
                num_of_frames_to_choose, gesture_dict_per_policy)

    def _random_balancing(self, number_of_frames_to_choose, gesture_dict):
        for gesture in gesture_dict:
            if len(gesture_dict[gesture]) > 0:
                selected_frames = random.choices(
                    gesture_dict[gesture], k=number_of_frames_to_choose)
                self.ballanced_data_set = self.ballanced_data_set + selected_frames

    def _parse_list_files(self, list_of_list_files):
        # depands only on csv files of splits directory
        for list_file in list_of_list_files:
            videos = [(x.strip().split(',')[0], x.strip().split(',')[1])
                      for x in open(list_file)]
            if self.debag:
                videos = [videos[0]]
            for video in videos:
                video_id = video[0]
                frame_count = int(video[1])

                gestures_file = os.path.join(
                    self.transcriptions_dir, video_id + ".txt")
                gestures = [[int(x.strip().split(' ')[0]), int(x.strip().split(' ')[1]), x.strip().split(' ')[2]]
                            for x in open(gestures_file)]
                # [start_frame, end_frame, gesture_id]

                _initial_labeled_frame = gestures[0][0]
                _final_labaled_frame = gestures[-1][1]
                _frame_count = frame_count

                _last_rgb_frame = os.path.join(self.root_path, video_id + self.video_suffix,
                                               'img_{:05d}.jpg'.format(_frame_count))

                self.frame_num_data[video_id] = [
                    _initial_labeled_frame, _final_labaled_frame]
                self._generate_labels_list(video_id, gestures)
                self._preload_images(
                    video_id, _frame_count, _final_labaled_frame)
                assert len(self.image_data[video_id]) == len(
                    self.labels_data[video_id])

    def _sort_frames_by_gesture(self, sampling_policy_index):
        gesture_dict = {}
        for gest in self.gesture_ids:
            if gest is None:
                continue
            gesture_dict[gest] = []
        real_snippet_length = self._real_length_calc(
            self.sampling_policies_list[sampling_policy_index])
        for video_id in self.labels_data:
            initial_relevant_frame_index = self.frame_num_data[video_id][0] + \
                real_snippet_length - 1
            frame_dict = {"video_name": video_id, "frame_index": 0,
                          "gesture": "", "sampling_policy_index": sampling_policy_index}
            for i in range(initial_relevant_frame_index, self.frame_num_data[video_id][1]):
                frame_dict["frame_index"] = i
                frame_dict["gesture"] = self.labels_data[video_id][i]
                gesture_dict[self.gesture_ids[self.labels_data[video_id][i]]].append(
                    frame_dict.copy())
        return gesture_dict

    def _generate_labels_list(self, video_id, gestures):
        labels_list = []
        for idx in range(gestures[0][0]-1):
            labels_list.append(None)

        for gesture in gestures:
            for idx in range(gesture[0], gesture[1]+1):
                labels_list.append(self.gesture_ids.index(gesture[2]))
        self.labels_data[video_id] = labels_list

    def _preload_images(self, video_id, _frame_count, _final_labaled_frame):
        print("Preloading images from video {}...".format(video_id))
        images = []
        img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        for idx in range(1, _final_labaled_frame + 1):
            imgs = self._load_image(img_dir, idx)
            images.extend(imgs)
        self.image_data[video_id] = images

    def _load_image(self, directory, idx):
        img = Image.open(os.path.join(
            directory, self.image_tmpl.format(idx))).convert('RGB')
        return [img]

    def _real_length_calc(self, policy_list):
        policy_arr = np.array(policy_list)
        assert len(policy_arr) + 1 == self.snippet_length
        return policy_arr.sum() + len(policy_arr) + 1

    def _get_the_min_numb_of_frames_per_calas(self, sampling_policy):
        gesture_hist = [0] * len(self.gesture_ids)

        real_snippet_length = self._real_length_calc(sampling_policy)
        tot = 0
        for video in self.labels_data:

            tot = tot + len(self.labels_data[video])
            for i, G_x in enumerate(self.gesture_ids):
                accessible_labels_data = self.labels_data[video][real_snippet_length +
                                                                 self.frame_num_data[video][0] - 1:-1]
                gesture_hist[i] = gesture_hist[i] + \
                    accessible_labels_data.count(self.gesture_ids.index(G_x))
        missing_gestures = []
        for j in range(1, len(gesture_hist)):
            if gesture_hist[j] == 0:
                missing_gestures.append(self.gesture_ids[j])
        if len(missing_gestures) > 0:
            print("Warning! Gestures missing in dataset for policy" +
                  str(sampling_policy), missing_gestures)
        gesture_hist = np.array(gesture_hist)
        the_minimum_number_of_frames = min(gesture_hist[gesture_hist > 0])
        return the_minimum_number_of_frames

    def __getitem__(self, index):
        snipt_info = self.ballanced_data_set[index]
        video_id = snipt_info["video_name"]
        idx = snipt_info["frame_index"]
        policy_idx = snipt_info["sampling_policy_index"]
        # label = snipt_info["gesture"]
        frame_list = self._policy_and_frame_into_frames_indices(
            idx, policy_idx)
        data = self.get_snippet(video_id, frame_list)

        target = self._get_snippet_labels(video_id, frame_list)
        # target_individual = torch.tensor(int(label[1]))

        return data, target

    def _get_snippet_labels(self, video_id, frame_list):
        assert self.return_dense_labels
        labels = self.labels_data[video_id]
        target = []
        for idx in frame_list:
            target.append(int(labels[idx]))
        return torch.tensor(target, dtype=torch.long)

    def _policy_and_frame_into_frames_indices(self, idx, policy_idx):
        policy = self.sampling_policies_list[policy_idx]
        frame_list = [idx]
        for dist in policy[::-1]:
            frame_list = [frame_list[0] - (dist + 1)] + frame_list
        return frame_list

    def get_snippet(self, video_id, frame_list):
        snippet = list()
        for idx in frame_list:
            _idx = max(idx, 0)  # padding if required
            img = self.image_data[video_id][_idx]
            snippet.append(img)
        snippet = rotate_snippet(snippet, 0.5)
        Add_Gaussian_Noise_to_snippet(snippet)
        snippet = self.transform(snippet)
        snippet = [torchvision.transforms.ToTensor()(img) for img in snippet]
        snippet = torch.stack(snippet, 0)
        snippet = self.normalize(snippet)
        snippet = snippet.view(((self.snippet_length,) + snippet.size()[-3:]))
        snippet = snippet.permute(1, 0, 2, 3)
        data = snippet
        return data

    def __len__(self):
        return len(self.ballanced_data_set)


class Sequential2DTestGestureDataSet(data.Dataset):

    def __init__(self, root_path, video_id, frame_count, transcriptions_dir, gesture_ids,
                 snippet_length=16, sampling_step=6,
                 image_tmpl='img_{:05d}.jpg', video_suffix="_capture2",
                 return_3D_tensor=True, return_dense_labels=True,
                 transform=None, normalize=None, preload=True):

        self.preload = preload
        self.root_path = root_path
        self.video_name = video_id
        self.video_id = video_id
        self.transcriptions_dir = transcriptions_dir
        self.gesture_ids = gesture_ids
        self.snippet_length = snippet_length
        self.sampling_step = sampling_step
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.return_3D_tensor = return_3D_tensor
        self.return_dense_labels = return_dense_labels
        self.transform = transform
        self.normalize = normalize
        self.frame_count = int(frame_count)

        self.gesture_sequence_per_video = {}
        self.image_data = {}
        self.frame_num_data = {}
        self.labels_data = {}
        self._parse_list_files(video_id)

    def _parse_list_files(self, video_name):
        # depands only on csv files of splits directory
        video_id = video_name
        _frame_count = self.frame_count

        gestures_file = os.path.join(
            self.transcriptions_dir, video_id + ".txt")
        gestures = [[int(x.strip().split(' ')[0]), int(x.strip().split(' ')[1]), x.strip().split(' ')[2]]
                    for x in open(gestures_file)]
        # [start_frame, end_frame, gesture_id]

        _initial_labeled_frame = gestures[0][0]
        _final_labaled_frame = gestures[-1][1]

        _last_rgb_frame = os.path.join(self.root_path, video_id + self.video_suffix,
                                       'img_{:05d}.jpg'.format(_frame_count))

        self.frame_num_data[video_id] = list(
            range(_initial_labeled_frame, _final_labaled_frame + 1, self.sampling_step))
        self._generate_labels_list(video_id, gestures)
        if self.preload:
            self._preload_images(video_id)
        assert len(self.image_data[video_id]) == len(
            self.labels_data[video_id])

    def _generate_labels_list(self, video_id, gestures):
        labels_list = []
        frame_nums_list = []
        img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        for frame_num in self.frame_num_data[self.video_name]:
            path = os.path.join(img_dir, self.image_tmpl.format(frame_num))
            if os.path.exists(path):
                frame_nums_list.append(frame_num)
                for gesture in gestures:
                    if frame_num >= gesture[0] and frame_num <= gesture[1]:
                        labels_list.append(self.gesture_ids.index(gesture[2]))
                        break
        # for gesture in gestures:
        #     for idx in range(gesture[0],gesture[1]+1):
        #         labels_list.append(self.gesture_ids.index(gesture[2]))
        self.labels_data[video_id] = labels_list
        self.image_data[video_id] = frame_nums_list

    def _preload_images(self, video_id):
        print("Preloading images from video {}...".format(video_id))
        images = []
        img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        for idx in self.frame_num_data[self.video_name]:
            try:
                imgs = self._load_image(img_dir, idx)
                images.extend(imgs)
            except FileNotFoundError:
                print(
                    f"{os.path.join(img_dir, self.image_tmpl.format(idx))} does not exist, skipped")

            self.image_data[video_id] = images

    def _load_image(self, directory, idx):
        img = Image.open(os.path.join(
            directory, self.image_tmpl.format(idx))).convert('RGB')
        return [img]

    def __getitem__(self, index):
        frame_list = [index]
        target = self._get_snippet_labels(self.video_name, frame_list)
        target = target[-1]
        data = self.get_snippet(self.video_name, index)

        return data, target

    def _get_snippet_labels(self, video_id, frame_list):
        assert self.return_dense_labels
        labels = self.labels_data[video_id]
        target = []
        idx = frame_list[-1]
        target.append(int(labels[idx]))
        return torch.tensor(target, dtype=torch.long)

    def get_snippet(self, video_id, idx):
        snippet = list()
        _idx = max(idx, 0)  # padding if required
        if self.preload:
            img = self.image_data[video_id][_idx]
        else:
            frame_num = self.image_data[video_id][_idx]
            mg_dir = os.path.join(
                self.root_path, video_id + self.video_suffix)
            img = self._load_image(
                mg_dir, frame_num)
            img = img[0]

        snippet.append(img)
        #snippet = rotate_snippet(snippet, 0.5)
        # Add_Gaussian_Noise_to_snippet(snippet)
        snippet = self.transform(snippet)
        snippet = [torchvision.transforms.ToTensor()(img)
                   for img in snippet]
        snippet = snippet[0]
        snippet = self.normalize(snippet)
        data = snippet
        return data

    def __len__(self):
        return (len(self.labels_data[self.video_name])) - (self.snippet_length - 1)


class Gesture2DTrainSet(data.Dataset):
    def __init__(self, root_path, list_of_list_files, transcriptions_dir, gesture_ids,
                 temporal_augmentaion_factor=0.2,
                 image_tmpl='img_{:05d}.jpg', video_suffix="_capture2",
                 transform=None, normalize=None, number_of_samples_per_class=400,
                 debag=False, preload=True):
        self.debag = debag
        self.root_path = root_path
        self.list_of_list_files = list_of_list_files
        self.transcriptions_dir = transcriptions_dir
        self.gesture_ids = gesture_ids
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.transform = transform
        self.normalize = normalize
        self.preload = preload
        self.gesture_sequence_per_video = {}
        self.image_data = {}
        self.frame_num_data = {}
        self.labels_data = {}
        self._parse_list_files(list_of_list_files)
        self.ballanced_data_set = []

        num_of_frames_to_choose = number_of_samples_per_class
        gesture_dict = self._sort_frames_by_gesture()
        self._random_balancing(num_of_frames_to_choose, gesture_dict)

        if self.preload:
            self._load_balanced_images()

    def _load_balanced_images(self):
        for snipt_info in self.ballanced_data_set:
            video_id = snipt_info["video_name"]
            idx = snipt_info["frame_index"]
            img = self.image_data[video_id][idx]
            if isinstance(img, Tuple):
                self.image_data[video_id][idx] = self._load_image(
                    *self.image_data[video_id][idx])[0]

    def _random_balancing(self, number_of_frames_to_choose, gesture_dict):

        if number_of_frames_to_choose == 0:
            for gesture in gesture_dict:
                self.ballanced_data_set += gesture_dict[gesture]
            return self.ballanced_data_set

        for gesture in gesture_dict:
            if len(gesture_dict[gesture]) > 0:
                selected_frames = random.choices(
                    gesture_dict[gesture], k=number_of_frames_to_choose)
                self.ballanced_data_set = self.ballanced_data_set + selected_frames

    def _parse_list_files(self, list_of_list_files):
        # depands only on csv files of splits directory
        for list_file in list_of_list_files:
            videos = [(x.strip().split(',')[0], x.strip().split(',')[1])
                      for x in open(list_file)]
            if self.debag:
                videos = [videos[0]]
            for video in videos:
                video_id = video[0]
                frame_count = int(video[1])

                gestures_file = os.path.join(
                    self.transcriptions_dir, video_id + ".txt")
                gestures = [[int(x.strip().split(' ')[0]), int(x.strip().split(' ')[1]), x.strip().split(' ')[2]]
                            for x in open(gestures_file)]
                # [start_frame, end_frame, gesture_id]

                _initial_labeled_frame = gestures[0][0]
                _final_labaled_frame = gestures[-1][1]
                _frame_count = frame_count

                _last_rgb_frame = os.path.join(self.root_path, video_id + self.video_suffix,
                                               'img_{:05d}.jpg'.format(_frame_count))

                self.frame_num_data[video_id] = [
                    _initial_labeled_frame, _final_labaled_frame]
                self._generate_labels_list(video_id, gestures)
                self._assert_images(video_id, _frame_count,
                                    _final_labaled_frame)
                assert len(self.image_data[video_id]) == len(
                    self.labels_data[video_id])

    def _sort_frames_by_gesture(self):
        gesture_dict = {}
        for gest in self.gesture_ids:
            if gest is None:
                continue
            gesture_dict[gest] = []
        real_snippet_length = 1
        for video_id in self.labels_data:
            initial_relevant_frame_index = self.frame_num_data[video_id][0] + \
                real_snippet_length - 1
            frame_dict = {"video_name": video_id,
                          "frame_index": 0, "gesture": ""}
            for i in range(initial_relevant_frame_index, self.frame_num_data[video_id][1]):
                frame_dict["frame_index"] = i
                frame_dict["gesture"] = self.labels_data[video_id][i]
                gesture_dict[self.gesture_ids[self.labels_data[video_id][i]]].append(
                    frame_dict.copy())
        return gesture_dict

    def _generate_labels_list(self, video_id, gestures):
        labels_list = []
        for idx in range(gestures[0][0]-1):
            labels_list.append(None)

        for gesture in gestures:
            for idx in range(gesture[0], gesture[1]+1):
                labels_list.append(self.gesture_ids.index(gesture[2]))
        self.labels_data[video_id] = labels_list

    def _assert_images(self, video_id, _frame_count, _final_labaled_frame):
        print("Asserting images from video {}...".format(video_id))
        images = []
        img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        for idx in range(1, _final_labaled_frame + 1):
            imgs = [(img_dir, idx)]
            images.extend(imgs)
        self.image_data[video_id] = images

    def _load_image(self, directory, idx):
        img = Image.open(os.path.join(
            directory, self.image_tmpl.format(idx))).convert('RGB')
        return [img]

    def _real_length_calc(self, policy_list):
        policy_arr = np.array(policy_list)
        assert len(policy_arr) + 1 == self.snippet_length
        return policy_arr.sum() + len(policy_arr) + 1

    def _get_the_min_numb_of_frames_per_calas(self, sampling_policy):
        gesture_hist = [0] * len(self.gesture_ids)

        real_snippet_length = self._real_length_calc(sampling_policy)
        tot = 0
        for video in self.labels_data:

            tot = tot + len(self.labels_data[video])
            for i, G_x in enumerate(self.gesture_ids):
                accessible_labels_data = self.labels_data[video][real_snippet_length +
                                                                 self.frame_num_data[video][0] - 1:-1]
                gesture_hist[i] = gesture_hist[i] + \
                    accessible_labels_data.count(self.gesture_ids.index(G_x))
        missing_gestures = []
        for j in range(1, len(gesture_hist)):
            if gesture_hist[j] == 0:
                missing_gestures.append(self.gesture_ids[j])
        if len(missing_gestures) > 0:
            print("Warning! Gestures missing in dataset for policy" +
                  str(sampling_policy), missing_gestures)
        gesture_hist = np.array(gesture_hist)
        the_minimum_number_of_frames = min(gesture_hist[gesture_hist > 0])
        return the_minimum_number_of_frames

    def __getitem__(self, index):
        res = None

        try:
            snipt_info = self.ballanced_data_set[index]
            video_id = snipt_info["video_name"]
            idx = snipt_info["frame_index"]
            label = snipt_info["gesture"]
            data = self.get_snippet(video_id, idx)

            target = label
            res = data, target
            # target_individual = torch.tensor(int(label[1]))
        except FileNotFoundError as e:
            print("skipped over an image because of the following error")
            print(e)

        return res

    def _get_snippet_labels(self, video_id, frame_list):
        assert self.return_dense_labels
        labels = self.labels_data[video_id]
        target = []
        for idx in frame_list:
            target.append(int(labels[idx]))
        return torch.tensor(target, dtype=torch.long)

    def _policy_and_frame_into_frames_indices(self, idx, policy_idx):
        policy = self.sampling_policies_list[policy_idx]
        frame_list = [idx]
        for dist in policy[::-1]:
            frame_list = [frame_list[0] - (dist + 1)] + frame_list
        return frame_list

    def get_snippet(self, video_id, idx):
        snippet = list()
        _idx = max(idx, 0)  # padding if required
        img = self.image_data[video_id][_idx]
        if not self.preload:
            img = self._load_image(*img)[0]
        snippet.append(img)
        snippet = rotate_snippet(snippet, 0.5)
        # Add_Gaussian_Noise_to_snippet(snippet)
        snippet = self.transform(snippet)
        snippet = [torchvision.transforms.ToTensor()(img) for img in snippet]
        snippet = snippet[0]
        snippet = self.normalize(snippet)
        data = snippet
        return data

    def __len__(self):
        return len(self.ballanced_data_set)


def rotate_snippet(snippet, max_angle):
    preform = random.uniform(0, 1)
    if preform > 0.5:
        new_snippet = []
        angle = random.uniform(-max_angle, max_angle)
        for img in snippet:
            new_img = rotate_img(img, angle)
            new_snippet.append(new_img)
        return new_snippet
    else:
        return snippet


def rotate_img(image, angle):
    image = np.array(image)
    num_rows, num_cols = (image.shape[:2])
    rotation_matrix = cv2.getRotationMatrix2D(
        (num_cols / 2, num_rows / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
    image = Image.fromarray(image)

    return image


def Add_Gaussian_Noise_to_snippet(snippet):
    preform = random.uniform(0, 1)
    if preform > 0.5:
        new_snippet = []
        sigma = random.uniform(0, 0.08)
        for img in snippet:
            new_img = Add_Gaussian_Noise(img, sigma)
            new_snippet.append(new_img)
        return new_snippet
    else:
        return snippet


def Add_Gaussian_Noise(image, sigma):
    image = np.array(image)
    noisyRandom = random_noise(image, var=sigma ** 2)
    im = Image.fromarray((noisyRandom * 255).astype(np.uint8))
    return im


def Pre_process_Kinematics(csv_path, select_data_type: list):
    """

    :param csv_path:
    :param select_data_type: list of length 4 that contains 1 ot 0 for [x,y,z_posision,x,y,z_velocity,euler_angle,angular_velocity ]
    only PSM (Patient side manipulator)!!
    :return:
    """
    table = read_txt_file(csv_path)
    print(table)
    relevant_coordinates = []
    if select_data_type[0] == 1:
        relevant_coordinates = relevant_coordinates + \
            [38, 39, 40, 38+19, 39+19, 40+19]

    relevant_columns = table.iloc[:, relevant_coordinates]
    print(relevant_columns)


def read_txt_file(csv_path):
    SPACE = " "
    file1 = open(csv_path, 'r')
    Lines = file1.readlines()
    table = []
    for line in Lines:
        line_list = []
        prev_char = SPACE
        current_string = ""
        for char in line:
            if char != SPACE:
                current_string = current_string + char
            elif prev_char != SPACE and char == SPACE:
                line_list.append(float(current_string))
                current_string = ""
            prev_char = char
        table.append(line_list)
    return pd.DataFrame(table)
