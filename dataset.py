#!/usr/bin/env python
import os
import sys
import numpy as np
import pickle
import json
import tables
import h5py
import queue
import time
import math

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores

def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def calc_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] -  inter[0]) / (union[1] - union[0])
    return iou


def calc_nIoL(base, sliding_clip):
    '''
    The reason we use nIoL is that we want the the most part of the sliding
    window clip to overlap with the assigned sentence, and simply increasing
    IoU threshold would harm regression layers ( regression aims to move the
    clip from low IoU to high IoU).
    '''
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1] - inter[0]
    sliding_l = sliding_clip[1] - sliding_clip[0]
    nIoL = 1.0 * (sliding_l - inter_l) / sliding_l
    return nIoL


class AnetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sliding_dir,
                 it_path,
                 visual_dim,
                 sentence_embed_dim,
                 IoU=0.5,
                 nIoU=0.15,
                 context_num=1,
                 context_size=128,
                ):
        self.sliding_dir = sliding_dir
        self.it_path = it_path
        self.visual_dim = visual_dim
        self.sentence_embed_dim = sentence_embed_dim
        self.IoU = IoU
        self.nIoU = nIoU
        self.context_num = context_num
        self.context_size = context_size

        #self.load_data()
        f = open('../data/vocab.json')
        self.vocab = json.loads(f.read())[0]
        self.mode = "train"
        self.window_size = 768#config.window_size

        self.mode = 'Train'
        if self.mode == 'Train':
            self.window_step = 128#config.window_step
        else:
            self.window_step = 256#config.inference_window_step

        self.feature_dim = 500#config.feature_dim
        self.sampels = []

        if self.mode == 'Train':
            self.data = read_json('../data/train.json')
        elif self.mode == 'Val':
            self.data = read_json('../data/val.json')

        elif self.mode == 'Valtrain':
            self.data = read_json('../data/val.json')

        elif self.mode == 'Test':
            self.data = read_json('../data/test.json')

        self.features_h5py = tables.open_file('../data/sub_activitynet_v1-3.c3d.hdf5', 'r')
        self._process_video()

        print('The number of {} dataset Activitynet samples is {}'.format(self.mode, len(self.sampels)))

    def load_data(self):
        '''
        Note:
            self.clip_sentence_pairs     : list of (ori_clip_name, sent_vec)
            self.clip_sentence_pairs_iou : list of (ori_clip_name, sent_vec, clip_name(with ".npy"), s_o, e_o) —— not all ground truth
        '''
        # movie_length_info = pickle.load(open("./video_allframes_info.pkl", 'rb'), encoding='iso-8859-1')
        print("Reading training data list from " + self.it_path)
        csv = pickle.load(open(self.it_path, 'rb'), encoding='iso-8859-1')
        self.clip_sentence_pairs = []
        for l in csv:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))

        movie_names_set = set()
        self.movie_clip_names = {}
        # read groundtruth sentence-clip pairs
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        self.num_samples = len(self.clip_sentence_pairs)
        print(str(len(self.clip_sentence_pairs))+" clip-sentence pairs are readed")
        
        # read sliding windows, and match them with the groundtruths to make training samples
        sliding_clips_tmp = os.listdir(self.sliding_dir)
        self.clip_sentence_pairs_iou = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2]=="npy":
                movie_name = clip_name.split("_")[0]
                for clip_sentence in self.clip_sentence_pairs:
                    original_clip_name = clip_sentence[0] 
                    original_movie_name = original_clip_name.split("_")[0]
                    if original_movie_name == movie_name:
                        start = int(clip_name.split("_")[1])
                        end = int(clip_name.split("_")[2].split(".")[0])
                        o_start = int(original_clip_name.split("_")[1]) 
                        o_end = int(original_clip_name.split("_")[2].split(".")[0])
                        iou = calc_IoU((start, end), (o_start, o_end))#筛选出train sample
                        if iou > self.IoU:
                            nIoL = calc_nIoL((o_start, o_end), (start, end))
                            if nIoL < self.nIoU:
                                # movie_length = movie_length_info[movie_name.split(".")[0]]
                                start_offset = o_start - start
                                end_offset = o_end - end
                                self.clip_sentence_pairs_iou.append((clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))
        self.num_samples_iou = len(self.clip_sentence_pairs_iou)
        print(str(len(self.clip_sentence_pairs_iou))+" iou clip-sentence pairs are readed")

    def __len__(self):
        #return self.num_samples_iou
        return len(self.sampels)

    def __getitem__(self, index):

        s = self.sampels[index]
        
        
        # read context features
        # left_context_feat, right_context_feat = self.get_context_window(self.clip_sentence_pairs_iou[index][2])
        # feat_path = os.path.join(self.sliding_dir, self.clip_sentence_pairs_iou[index][2])
        # featmap = np.load(feat_path)
        # vis = np.hstack((left_context_feat, featmap, right_context_feat))
        #
        # sent = self.clip_sentence_pairs_iou[index][1][:self.sentence_embed_dim]
        #
        # p_offset = self.clip_sentence_pairs_iou[index][3]
        # l_offset = self.clip_sentence_pairs_iou[index][4]
        # offset = np.array([p_offset, l_offset], dtype=np.float32)
        bs = list(self.sampels[index][0].size())[0]
        data_torch = {
            'vis': self.sampels[index][0].reshape(bs, -1),
            'sent': self.sampels[index][2],
            'offset': self.sampels[index][1],
        }
        return data_torch


    def get_context_window(self, clip_name): #只取出上下文的feat组成feats，然后求均值
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        left_context_feats = np.zeros([self.context_num, self.visual_dim // 3], dtype=np.float32)
        right_context_feats = np.zeros([self.context_num, self.visual_dim // 3], dtype=np.float32)
        last_left_feat = np.load(os.path.join(self.sliding_dir, clip_name))
        last_right_feat = np.load(os.path.join(self.sliding_dir, clip_name))
        for k in range(self.context_num):
            left_context_start = start - self.context_size * (k + 1)
            left_context_end = start - self.context_size * k
            right_context_start = end + self.context_size * k
            right_context_end = end + self.context_size * (k + 1)
            left_context_name = movie_name + "_" + str(left_context_start) + "_" + str(left_context_end) + ".npy"
            right_context_name = movie_name + "_" + str(right_context_start) + "_" + str(right_context_end) + ".npy"

            left_context_path = os.path.join(self.sliding_dir, left_context_name)
            if os.path.exists(left_context_path):
                left_context_feat = np.load(left_context_path)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat

            right_context_path = os.path.join(self.sliding_dir, right_context_name)
            if os.path.exists(right_context_path):
                right_context_feat = np.load(right_context_path)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat #不存在则用上一次检索结果替代

            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

    def _process_video(self):  # 6*feature
        count = 0
        for video_name in self.data.keys():
            count += 1
            if count > 100 and self.mode == 'Val':
                break
            if count > 30 and self.mode == 'Test':
                break
            d = self.data[video_name]
            sentences = d['sentences']
            # data = torch.Tensor(self.features[video_name])
            data = torch.Tensor(self.features_h5py.root[video_name]['c3d_features'].read())
            times = d['timestamps']
            length = data.shape[0]
            ratio = float(length) / float(d['duration'])  # ~3.7,3.8
            length = int(d['duration'] * ratio)

            pair = []
            for sentence, time in zip(sentences, times):
                word_indexes = []
                for word in sentence.split(' '):
                    idx = self.vocab.get(word)

                    if idx is not None:
                        word_indexes.append(idx)
                    else:
                        word_indexes.append(0)
                sent = np.array(word_indexes, dtype=np.long)
                gt_start = float(time[0])
                gt_end = float(time[1] + 1.)
                if gt_start > gt_end:
                    gt_start, gt_end = gt_end, gt_start
                gt = [np.round(gt_start * ratio), np.round(gt_end * ratio)]
                pair.append([sent, gt])

            window_size, stride = self.window_size, self.window_step
            n_window = math.ceil((length + stride - window_size) / stride)
            windows_start = [i * stride for i in range(int(n_window))]
            if length < window_size:
                windows_start = [0]

            elif length - windows_start[-1] - window_size > 0:
                windows_start.append(length - window_size)

            for start in windows_start:
                # tmp_data = _get_feature(self.feature_path, video_name, start, start+window_size)
                window = min(window_size, length - start)
                pad_dim = window_size + start - length
                tmp_data = data[start:start + window, :]  # [window_size, feature_dim]
                if pad_dim > 0:
                    pad = torch.zeros(pad_dim, self.feature_dim)
                    tmp_data = torch.cat((tmp_data, pad), 0)

                xmin = start
                xmax = start + window_size

                for idx in range(len(pair)):
                    sent = pair[idx][0]
                    round_gt_start = pair[idx][1][0]
                    round_gt_end = pair[idx][1][1]

                    tmp_ioa = ioa_with_anchors(round_gt_start, round_gt_end, xmin, xmax)

                    # 有gt的片段被送去训练 test时也只送有交集的框 todo fix 另外测试无的部分
                    if tmp_ioa > 0:
                        # gt bbox info
                        corrected_start = max(round_gt_start, xmin) - xmin
                        corrected_end = min(round_gt_end, xmax) - xmin
                        # [0, window_size]
                        corrected_start = max(corrected_start, 0.0)
                        corrected_start = min(corrected_start, self.window_size)
                        corrected_end = max(corrected_end, 0.0)
                        corrected_end = min(corrected_end, self.window_size)

                        tmp_gt_bbox = [float(corrected_start), float(corrected_end)]
                        # tmp_gt_bbox = [float(corrected_start) / self.window_size,
                        #            float(corrected_end) / self.window_size]
                        # elif tmp_ioa <= 0:
                        #     tmp_gt_bbox = [0., 0.]
                        #     tmp_ioa = 0.

                        tmp_results = [tmp_data, np.array(tmp_gt_bbox),
                                       np.array(sent), np.array([round_gt_start, round_gt_end])]

                        if self.mode != 'Train':
                            # print(xmin, tmp_gt_bbox, round_gt_start, round_gt_end)
                            tmp_results.append(video_name)
                            tmp_results.append(sentences[idx])
                            tmp_results.append(xmin)
                            tmp_results.append(ratio)
                        self.sampels.append(tmp_results)

        self.features_h5py.close()


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sliding_dir,
                 it_path,
                 visual_dim,
                 sentence_embed_dim,
                 IoU=0.5,
                 nIoU=0.15,
                 context_num=1,
                 context_size=128,
                 ):
        self.sliding_dir = sliding_dir
        self.it_path = it_path
        self.visual_dim = visual_dim
        self.sentence_embed_dim = sentence_embed_dim
        self.IoU = IoU
        self.nIoU = nIoU
        self.context_num = context_num
        self.context_size = context_size

        self.load_data()

    def load_data(self):
        '''
        Note:
            self.clip_sentence_pairs     : list of (ori_clip_name, sent_vec)
            self.clip_sentence_pairs_iou : list of (ori_clip_name, sent_vec, clip_name(with ".npy"), s_o, e_o) —— not all ground truth
        '''
        # movie_length_info = pickle.load(open("./video_allframes_info.pkl", 'rb'), encoding='iso-8859-1')
        print("Reading training data list from " + self.it_path)
        csv = pickle.load(open(self.it_path, 'rb'), encoding='iso-8859-1')
        self.clip_sentence_pairs = []
        for l in csv:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))

        movie_names_set = set()
        self.movie_clip_names = {}
        # read groundtruth sentence-clip pairs
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        self.num_samples = len(self.clip_sentence_pairs)
        print(str(len(self.clip_sentence_pairs)) + " clip-sentence pairs are readed")

        # read sliding windows, and match them with the groundtruths to make training samples
        sliding_clips_tmp = os.listdir(self.sliding_dir)
        self.clip_sentence_pairs_iou = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2] == "npy":
                movie_name = clip_name.split("_")[0]
                for clip_sentence in self.clip_sentence_pairs:
                    original_clip_name = clip_sentence[0]
                    original_movie_name = original_clip_name.split("_")[0]
                    if original_movie_name == movie_name:
                        start = int(clip_name.split("_")[1])
                        end = int(clip_name.split("_")[2].split(".")[0])
                        o_start = int(original_clip_name.split("_")[1])
                        o_end = int(original_clip_name.split("_")[2].split(".")[0])
                        iou = calc_IoU((start, end), (o_start, o_end))
                        if iou > self.IoU:
                            nIoL = calc_nIoL((o_start, o_end), (start, end))
                            if nIoL < self.nIoU:
                                # movie_length = movie_length_info[movie_name.split(".")[0]]
                                start_offset = o_start - start
                                end_offset = o_end - end
                                self.clip_sentence_pairs_iou.append(
                                    (clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))
        self.num_samples_iou = len(self.clip_sentence_pairs_iou)
        print(str(len(self.clip_sentence_pairs_iou)) + " iou clip-sentence pairs are readed")

    def __len__(self):
        return self.num_samples_iou

    def __getitem__(self, index):
        # read context features
        left_context_feat, right_context_feat = self.get_context_window(self.clip_sentence_pairs_iou[index][2])
        feat_path = os.path.join(self.sliding_dir, self.clip_sentence_pairs_iou[index][2])
        featmap = np.load(feat_path)
        vis = np.hstack((left_context_feat, featmap, right_context_feat))

        sent = self.clip_sentence_pairs_iou[index][1][:self.sentence_embed_dim]

        p_offset = self.clip_sentence_pairs_iou[index][3]
        l_offset = self.clip_sentence_pairs_iou[index][4]
        offset = np.array([p_offset, l_offset], dtype=np.float32)

        data_torch = {
            'vis': torch.from_numpy(vis),
            'sent': torch.from_numpy(sent),
            'offset': torch.from_numpy(offset),
        }
        return data_torch

    def get_context_window(self, clip_name):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        left_context_feats = np.zeros([self.context_num, self.visual_dim // 3], dtype=np.float32)
        right_context_feats = np.zeros([self.context_num, self.visual_dim // 3], dtype=np.float32)
        last_left_feat = np.load(os.path.join(self.sliding_dir, clip_name))
        last_right_feat = np.load(os.path.join(self.sliding_dir, clip_name))
        for k in range(self.context_num):
            left_context_start = start - self.context_size * (k + 1)
            left_context_end = start - self.context_size * k
            right_context_start = end + self.context_size * k
            right_context_end = end + self.context_size * (k + 1)
            left_context_name = movie_name + "_" + str(left_context_start) + "_" + str(left_context_end) + ".npy"
            right_context_name = movie_name + "_" + str(right_context_start) + "_" + str(right_context_end) + ".npy"

            left_context_path = os.path.join(self.sliding_dir, left_context_name)
            if os.path.exists(left_context_path):
                left_context_feat = np.load(left_context_path)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat

            right_context_path = os.path.join(self.sliding_dir, right_context_name)
            if os.path.exists(right_context_path):
                right_context_feat = np.load(right_context_path)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat

            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)


class TestingAnetDataset(object):
    def __init__(self, img_dir, csv_path, batch_size):
        # il_path: image_label_file path
        # self.index_in_epoch = 0
        # self.epochs_completed = 0

        # self.load_data()
        f = open('../data/vocab.json')
        self.vocab = json.loads(f.read())[0]
        self.mode = "train"
        self.window_size = 768  # config.window_size

        self.mode = 'Test'
        if self.mode == 'Train':
            self.window_step = 128  # config.window_step
        else:
            self.window_step = 256  # config.inference_window_step

        self.feature_dim = 500  # config.feature_dim
        self.sampels = []

        if self.mode == 'Train':
            self.data = read_json('../data/train.json')
        elif self.mode == 'Val':
            self.data = read_json('../data/val.json')

        elif self.mode == 'Valtrain':
            self.data = read_json('../data/val.json')

        elif self.mode == 'Test':
            self.data = read_json('../data/test.json')

        self.features_h5py = tables.open_file('../data/sub_activitynet_v1-3.c3d.hdf5', 'r')
        self._process_video()

        print('The number of {} dataset Activitynet samples is {}'.format(self.mode, len(self.sampels)))


        self.batch_size = batch_size
        # self.image_dir = img_dir
        # print("Reading testing data list from " + csv_path)
        # self.semantic_size = 4800
        # csv = pickle.load(open(csv_path, 'rb'), encoding='iso-8859-1')
        # self.clip_sentence_pairs = []
        # for l in csv:
        #     clip_name = l[0]
        #     sent_vecs = l[1]
        #     for sent_vec in sent_vecs:
        #         self.clip_sentence_pairs.append((clip_name, sent_vec))
        # print(str(len(self.clip_sentence_pairs)) + " pairs are readed")
        # movie_names_set = set()
        # self.movie_clip_names = {}
        # for k in range(len(self.clip_sentence_pairs)):
        #     clip_name = self.clip_sentence_pairs[k][0]
        #     movie_name = clip_name.split("_")[0]
        #     if not movie_name in movie_names_set:
        #         movie_names_set.add(movie_name)
        #         self.movie_clip_names[movie_name] = []
        #     self.movie_clip_names[movie_name].append(k)
        # self.movie_names = list(movie_names_set)
        #
        # self.clip_num_per_movie_max = 0
        # for movie_name in self.movie_clip_names:
        #     if len(self.movie_clip_names[movie_name]) > self.clip_num_per_movie_max: self.clip_num_per_movie_max = len(
        #         self.movie_clip_names[movie_name])
        # print("Max number of clips in a movie is " + str(self.clip_num_per_movie_max))
        #
        # self.sliding_clip_path = img_dir
        # sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        # self.sliding_clip_names = []
        # for clip_name in sliding_clips_tmp:
        #     if clip_name.split(".")[2] == "npy":
        #         movie_name = clip_name.split("_")[0]
        #         if movie_name in self.movie_clip_names:
        #             self.sliding_clip_names.append(clip_name.split(".")[0] + "." + clip_name.split(".")[1])
        # self.num_samples = len(self.clip_sentence_pairs)
        # print("sliding clips number: " + str(len(self.sliding_clip_names)))
        # assert self.batch_size <= self.num_samples

    def _process_video(self):  # 6*feature
        count = 0
        for video_name in self.data.keys():
            count += 1
            if count > 100 and self.mode == 'Val':
                break
            if count > 30 and self.mode == 'Test':
                break
            d = self.data[video_name]
            sentences = d['sentences']
            # data = torch.Tensor(self.features[video_name])
            data = torch.Tensor(self.features_h5py.root[video_name]['c3d_features'].read())
            times = d['timestamps']
            length = data.shape[0]
            ratio = float(length) / float(d['duration'])  # ~3.7,3.8
            length = int(d['duration'] * ratio)

            pair = []
            for sentence, time in zip(sentences, times):
                word_indexes = []
                for word in sentence.split(' '):
                    idx = self.vocab.get(word)

                    if idx is not None:
                        word_indexes.append(idx)
                    else:
                        word_indexes.append(0)
                sent = np.array(word_indexes, dtype=np.long)
                gt_start = float(time[0])
                gt_end = float(time[1] + 1.)
                if gt_start > gt_end:
                    gt_start, gt_end = gt_end, gt_start
                gt = [np.round(gt_start * ratio), np.round(gt_end * ratio)]
                pair.append([sent, gt])

            window_size, stride = self.window_size, self.window_step
            n_window = math.ceil((length + stride - window_size) / stride)
            windows_start = [i * stride for i in range(int(n_window))]
            if length < window_size:
                windows_start = [0]

            elif length - windows_start[-1] - window_size > 0:
                windows_start.append(length - window_size)

            for start in windows_start:
                # tmp_data = _get_feature(self.feature_path, video_name, start, start+window_size)
                window = min(window_size, length - start)
                pad_dim = window_size + start - length
                tmp_data = data[start:start + window, :]  # [window_size, feature_dim]
                if pad_dim > 0:
                    pad = torch.zeros(pad_dim, self.feature_dim)
                    tmp_data = torch.cat((tmp_data, pad), 0)

                xmin = start
                xmax = start + window_size

                for idx in range(len(pair)):
                    sent = pair[idx][0]
                    round_gt_start = pair[idx][1][0]
                    round_gt_end = pair[idx][1][1]

                    tmp_ioa = ioa_with_anchors(round_gt_start, round_gt_end, xmin, xmax)

                    # 有gt的片段被送去训练 test时也只送有交集的框 todo fix 另外测试无的部分
                    if tmp_ioa > 0:
                        # gt bbox info
                        corrected_start = max(round_gt_start, xmin) - xmin
                        corrected_end = min(round_gt_end, xmax) - xmin
                        # [0, window_size]
                        corrected_start = max(corrected_start, 0.0)
                        corrected_start = min(corrected_start, self.window_size)
                        corrected_end = max(corrected_end, 0.0)
                        corrected_end = min(corrected_end, self.window_size)

                        tmp_gt_bbox = [float(corrected_start), float(corrected_end)]
                        # tmp_gt_bbox = [float(corrected_start) / self.window_size,
                        #            float(corrected_end) / self.window_size]
                        # elif tmp_ioa <= 0:
                        #     tmp_gt_bbox = [0., 0.]
                        #     tmp_ioa = 0.

                        tmp_results = [tmp_data, np.array(tmp_gt_bbox),
                                       np.array(sent), np.array([round_gt_start, round_gt_end])]

                        if self.mode != 'Train':
                            # print(xmin, tmp_gt_bbox, round_gt_start, round_gt_end)
                            tmp_results.append(video_name)
                            tmp_results.append(sentences[idx])
                            tmp_results.append(xmin)
                            tmp_results.append(ratio)
                        self.sampels.append(tmp_results)

        self.features_h5py.close()

    def get_clip_sample(self, sample_num, movie_name, clip_name):
        length = len(os.listdir(self.image_dir + movie_name + "/" + clip_name))
        sample_step = 1.0 * length / sample_num
        sample_pos = np.floor(sample_step * np.array(range(sample_num)))
        sample_pos_str = []
        img_names = os.listdir(self.image_dir + movie_name + "/" + clip_name)
        # sort is very important! to get a correct sequence order
        img_names.sort()
        # print img_names
        for pos in sample_pos:
            sample_pos_str.append(self.image_dir + movie_name + "/" + clip_name + "/" + img_names[int(pos)])
        return sample_pos_str

    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        clip_length = 128  # end-start
        left_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path + clip_name)
        last_right_feat = np.load(self.sliding_clip_path + clip_name)
        for k in range(win_length):
            left_context_start = start - clip_length * (k + 1)
            left_context_end = start - clip_length * k
            right_context_start = end + clip_length * k
            right_context_end = end + clip_length * (k + 1)
            left_context_name = movie_name + "_" + str(left_context_start) + "_" + str(left_context_end) + ".npy"
            right_context_name = movie_name + "_" + str(right_context_start) + "_" + str(right_context_end) + ".npy"
            if os.path.exists(self.sliding_clip_path + left_context_name):
                left_context_feat = np.load(self.sliding_clip_path + left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path + right_context_name):
                right_context_feat = np.load(self.sliding_clip_path + right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

    def load_movie(self, movie_name):
        movie_clip_sentences = []
        for k in range(len(self.clip_names)):
            if movie_name in self.clip_names[k]:
                movie_clip_sentences.append((self.clip_names[k], self.sent_vecs[k][:2400], self.sentences[k]))

        movie_clip_imgs = []
        for k in range(len(self.movie_frames[movie_name])):
            # print str(k)+"/"+str(len(self.movie_frames[movie_name]))
            if os.path.isfile(self.movie_frames[movie_name][k][1]) and os.path.getsize(
                    self.movie_frames[movie_name][k][1]) != 0:
                img = load_image(self.movie_frames[movie_name][k][1])
                movie_clip_imgs.append((self.movie_frames[movie_name][k][0], img))

        return movie_clip_imgs, movie_clip_sentences

    def load_movie_byclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []
        clip_set = set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append(
                    (self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))

                if not self.clip_sentence_pairs[k][0] in clip_set:
                    clip_set.add(self.clip_sentence_pairs[k][0])
                    # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                    visual_feature_path = self.image_dir + self.clip_sentence_pairs[k][0] + ".npy"
                    feature_data = np.load(visual_feature_path)
                    movie_clip_featmap.append((self.clip_sentence_pairs[k][0], feature_data))
        return movie_clip_featmap, movie_clip_sentences

    def load_movie_slidingclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []
        clip_set = set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append(
                    (self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))
        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                visual_feature_path = self.sliding_clip_path + self.sliding_clip_names[k] + ".npy"
                # context_feat=self.get_context(self.sliding_clip_names[k]+".npy")
                left_context_feat, right_context_feat = self.get_context_window(self.sliding_clip_names[k] + ".npy", 1)
                feature_data = np.load(visual_feature_path)
                # comb_feat=np.hstack((context_feat,feature_data))
                comb_feat = np.hstack((left_context_feat, feature_data, right_context_feat))
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))
        return movie_clip_featmap, movie_clip_sentences


class TestingDataSet(object):
    def __init__(self, img_dir, csv_path, batch_size):
        #il_path: image_label_file path
        #self.index_in_epoch = 0
        #self.epochs_completed = 0
        self.batch_size = batch_size
        self.image_dir = img_dir
        print("Reading testing data list from "+csv_path)
        self.semantic_size = 4800
        csv = pickle.load(open(csv_path, 'rb'), encoding='iso-8859-1')
        self.clip_sentence_pairs = []
        for l in csv:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))
        print(str(len(self.clip_sentence_pairs))+" pairs are readed")
        movie_names_set = set()
        self.movie_clip_names = {}
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        
        self.clip_num_per_movie_max = 0
        for movie_name in self.movie_clip_names:
            if len(self.movie_clip_names[movie_name])>self.clip_num_per_movie_max: self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])
        print("Max number of clips in a movie is "+str(self.clip_num_per_movie_max))
        
        self.sliding_clip_path = img_dir
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.sliding_clip_names = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2]=="npy":
                movie_name = clip_name.split("_")[0]
                if movie_name in self.movie_clip_names:
                    self.sliding_clip_names.append(clip_name.split(".")[0]+"."+clip_name.split(".")[1])
        self.num_samples = len(self.clip_sentence_pairs)
        print("sliding clips number: "+str(len(self.sliding_clip_names)))
        assert self.batch_size <= self.num_samples
        

    def get_clip_sample(self, sample_num, movie_name, clip_name):
        length=len(os.listdir(self.image_dir+movie_name+"/"+clip_name))
        sample_step=1.0*length/sample_num
        sample_pos=np.floor(sample_step*np.array(range(sample_num)))
        sample_pos_str=[]
        img_names=os.listdir(self.image_dir+movie_name+"/"+clip_name)
        # sort is very important! to get a correct sequence order
        img_names.sort()
       # print img_names
        for pos in sample_pos:
            sample_pos_str.append(self.image_dir+movie_name+"/"+clip_name+"/"+img_names[int(pos)])
        return sample_pos_str
    
    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        clip_length = 128#end-start
        left_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"_"+str(left_context_end)+".npy"
            right_context_name = movie_name+"_"+str(right_context_start)+"_"+str(right_context_end)+".npy"
            if os.path.exists(self.sliding_clip_path+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)


    def load_movie(self, movie_name):
        movie_clip_sentences=[]
        for k in range(len(self.clip_names)):
            if movie_name in self.clip_names[k]:
                movie_clip_sentences.append((self.clip_names[k], self.sent_vecs[k][:2400], self.sentences[k]))

        movie_clip_imgs=[]
        for k in range(len(self.movie_frames[movie_name])):
           # print str(k)+"/"+str(len(self.movie_frames[movie_name]))            
            if os.path.isfile(self.movie_frames[movie_name][k][1]) and os.path.getsize(self.movie_frames[movie_name][k][1])!=0:
                img=load_image(self.movie_frames[movie_name][k][1])
                movie_clip_imgs.append((self.movie_frames[movie_name][k][0],img))
                    
        return movie_clip_imgs, movie_clip_sentences

    def load_movie_byclip(self,movie_name,sample_num):
        movie_clip_sentences=[]
        movie_clip_featmap=[]
        clip_set=set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0],self.clip_sentence_pairs[k][1][:self.semantic_size]))

                if not self.clip_sentence_pairs[k][0] in clip_set:
                    clip_set.add(self.clip_sentence_pairs[k][0])
                    # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                    visual_feature_path=self.image_dir+self.clip_sentence_pairs[k][0]+".npy"
                    feature_data=np.load(visual_feature_path)
                    movie_clip_featmap.append((self.clip_sentence_pairs[k][0],feature_data))
        return movie_clip_featmap, movie_clip_sentences
    
    def load_movie_slidingclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []
        clip_set = set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))
        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                visual_feature_path = self.sliding_clip_path+self.sliding_clip_names[k]+".npy"
                #context_feat=self.get_context(self.sliding_clip_names[k]+".npy")
                left_context_feat,right_context_feat = self.get_context_window(self.sliding_clip_names[k]+".npy",1)
                feature_data = np.load(visual_feature_path)
                #comb_feat=np.hstack((context_feat,feature_data))
                comb_feat = np.hstack((left_context_feat,feature_data,right_context_feat))
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))
        return movie_clip_featmap, movie_clip_sentences

