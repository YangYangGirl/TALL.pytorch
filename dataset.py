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

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import threading

class WordEmbedding(nn.Module):
    """Word Embedding
    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout=0):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        #self.emb.weight.data = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class SentenceEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(SentenceEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch, seq_len, _ = x.shape
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        # if self.ndirections == 1:
        #     return output
        # forward_ = output[:, -1, :self.num_hid]
        # index = [seq_len-1-i for i in range(seq_len)]
        # backward = output[:, :, self.num_hid:][:,index,:]
        # return torch.cat((forward_, backward), dim=2)   #N, seq_len, hid*2
        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output

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


class CharadesDataset(torch.utils.data.Dataset):

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
        f = open('../data/vocab.json')
        self.vocab = json.loads(f.read())[0]
        self.feature_path = "../data/Charades_v1_features_rgb"
        self.unit_size = 1
        self.feature_dim = 4096
        self.pkl_path = "../data/Charades_feature_rgb_pkl"
        self.window_size = 192
        self.window_step = 64
        self.num_classes = 2  # action categroies + BG for THUMOS14 is 21
        self.mode = 'Train'
        self.sampels = []
        if self.mode == 'Train':
            self.clip_gt_path = "../data/charades_sta_train.txt"
        elif self.mode == 'Val':
            self.clip_gt_path = "../data/charades_sta_train.txt"
            self.window_step =  64
        elif self.mode == 'Valtrain':
            self.clip_gt_path = "../data/charades_sta_train.txt"
            self.window_step = 64
        elif self.mode == 'Test':
            self.clip_gt_path = "../data/charades_sta_test.txt"
            self.window_step = 64
        # multi thread
        self.exit_flag = False
        self.queue_lock = threading.Lock()
        self.queue = queue.Queue()
        self.item_done = 0
        self._preparedata()
        print(
            'The number of {} dataset samples is {}'.format(self.mode, len(self.sampels)))

    def _preparedata(self):

        print('wait...prepare data')
        window_size = self.window_size
        stride = self.window_step
        id = 0

        # multithread
        for _ in range(8):
            thread = threading.Thread(target=self.process_thread, args=[self])
            thread.start()
        last_name = ''
        with open(self.clip_gt_path) as f:
            self.queue_lock.acquire()
            l_list = []
            for l in f:
                video_name = l.rstrip().split(" ")[0]
                if video_name == last_name or last_name == '':
                    l_list.append(l)
                else:
                    if len(l_list):
                        self.queue.put(l_list)
                    l_list = [l]
                last_name = video_name
                id = id + 1
            if len(l_list):
                self.queue.put(l_list)
            self.queue_lock.release()

        while not self.queue.empty():
            last = self.item_done
        #            time.sleep(10)
        # if self.item_done % 100 == 0:
        #     print(self.item_done, (self.item_done - last) / 10)
        # if self.item_done > 10 and self.mode=='Val':
        #    break
        # if self.item_done > 100:
        #    break

        self.exit_flag = True

    #                if id%10==0:
    #                    print(id)

    def _get_feature(self, video_name, start, end):
        feat_dir = self.feature_path
        swin_step = 4
        all_feat = np.zeros([0, self.feature_dim], dtype=np.float32)
        current_pos = start * 4
        end = (end - 1) * 4
        feat_path = os.path.join(feat_dir, video_name)

        pkl_path = self.pkl_path
        # if os.path.exists(pkl_path) == False:
        #     os.mkdir(pkl_path)
        pickle_path = os.path.join(pkl_path, video_name + '_f.pkl')

        if os.path.exists(pickle_path):
            p_f = open(pickle_path, 'rb')
            all_feat = pickle.load(p_f)
            p_f.close()
            return all_feat
        while current_pos < end:
            clip_name = video_name + '-' + str(current_pos + 1).zfill(6) + '.txt'
            path = os.path.join(feat_path, clip_name)
            if os.path.exists(path):
                feat = np.loadtxt(path)
            else:
                # print(video_name,current_pos + 1)
                feat = np.zeros(self.feature_dim, dtype=np.float32)
            all_feat = np.vstack((all_feat, feat))
            current_pos += swin_step

        # print(all_feat.shape)
        # time.sleep(100)

        p_f = open(pickle_path, 'wb')
        pickle.dump(all_feat, p_f)
        p_f.close()
        return all_feat  # window_size*4096

    def process_thread(sel, self):
        while not self.exit_flag:
            self.queue_lock.acquire()
            if not self.queue.empty():
                l = self.queue.get()
                self.item_done += 1
                self.queue_lock.release()
                self.process_video(l)
            else:
                self.queue_lock.release()

    def process_video(self, l_list):
        window_size, stride = self.window_size, self.window_step
        l_length = len(l_list)
        l = l_list[0]
        video_name = l.rstrip().split(" ")[0]
        clip_list = os.listdir(self.feature_path + '/' + video_name)
        length = len(clip_list)  # temporal length]
        video_feature_total = self._get_feature(video_name, 0, length)
        video_feature_total = torch.Tensor(video_feature_total)

        for id in range(l_length):
            l = l_list[id]
            gt_start = float(l.rstrip().split(" ")[1])
            gt_end = float(l.rstrip().split(" ")[2].split("##")[0])
            sentence = l.rstrip().split("##")[1][:-1]

            if gt_start > gt_end:
                gt_start, gt_end = gt_end, gt_start

            round_gt_start = round(gt_start * 24 / 4) * 4
            round_gt_end = round(gt_end * 24 / 4) * 4
            n_window = math.ceil((length + stride - window_size) / stride)
            # print(n_window, length)
            frameList = [1 + self.unit_size * i for i in range(length)]  # 1, 2, 3
            windows_start = [i * stride for i in range(int(n_window))]

            word_indexes = []
            for word in sentence.split(' '):
                idx = self.vocab.get(word)
                if idx is not None:
                    word_indexes.append(idx)
                else:
                    word_indexes.append(0)  # <UNK>
            sent = np.array(word_indexes, dtype=np.long)
            flag = 1
            if length < window_size:
                windows_start = [0]
                frameList.extend([frameList[-1] + self.unit_size * (i + 1) for i in range(window_size - length)])

            elif length - windows_start[-1] - window_size > 0:
                flag = 0
                windows_start.append(length - window_size)
            xmin = [(i - 1) * 4 + 1 for i in frameList]
            xmax = frameList[1:]
            xmax.append(frameList[-1] + 1)
            xmax = [(i - 1) * 4 + 1 for i in xmax]

            for start in windows_start:
                # tmp_data = _get_feature(self.feature_path, video_name, start, start+window_size)
                window = min(window_size, video_feature_total.shape[0] - start)
                pad_dim = window_size + start - video_feature_total.shape[0]
                tmp_data = video_feature_total[start:start + window, :]  # [window_size, feature_dim]
                if pad_dim > 0:
                    pad = torch.zeros(pad_dim, self.feature_dim)
                    tmp_data = torch.cat((tmp_data, pad), 0)
                # if(tmp_data.shape[0] != 128):
                #    print(video_name, sentence,start, start + window_size,video_feature_total.shape)
                #    time.sleep(1000000)
                tmp_anchor_xmins = xmin[start:start + window_size]
                tmp_anchor_xmaxs = xmax[start:start + window_size]
                # gt_box -iou;   label:0/1;    feat; query
                tmp_ioa = ioa_with_anchors(round_gt_start, round_gt_end, tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1])
                if tmp_ioa > 0:
                    # gt bbox info
                    corrected_start = max(round_gt_start, tmp_anchor_xmins[0]) - tmp_anchor_xmins[0]
                    corrected_end = min(round_gt_end, tmp_anchor_xmaxs[-1]) - tmp_anchor_xmins[0]

                    corrected_start = max(corrected_start, 0.0)
                    corrected_start = min(corrected_start, self.window_size * 4)
                    corrected_end = max(corrected_end, 0.0)
                    corrected_end = min(corrected_end, self.window_size * 4)

                    tmp_gt_bbox = [int(float(corrected_start) / 4),
                                   int(float(corrected_end) / 4)]
                else:
                    continue
                label = 0
                if (tmp_ioa > 0 and flag == 0 and tmp_anchor_xmins[0] == xmin[0]):
                    label = 1
                ###Todo add sample balance : label = 1 positive; label = 0 negative

                #                        print(video_name, round_gt_start, round_gt_end, tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1],tmp_gt_bbox, label,  tmp_data.shape, tmp_ioa)
                # the overlap region is corrected
                tmp_results = [tmp_data, np.array(tmp_gt_bbox), np.array(sent),
                               np.array([round_gt_start, round_gt_end])]
                if self.mode != 'Train':
                    tmp_results.append(video_name)
                    tmp_results.append(sentence)
                    tmp_results.append(tmp_anchor_xmins[0])
                # print(video_name, sentence, tmp_data.shape, tmp_ioa, [round_gt_start, round_gt_end],[tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1]],tmp_gt_bbox)
                # time.sleep(10)
                self.sampels.append(tmp_results)

    def __getitem__(self, index):

        s = self.samples[index]
        sent_raw = self.samples[index][2]

        data_torch = {
            'vis': self.samples[index][0],  # .reshape(bs, -1),
            'sent': sent_raw,
            'offset': self.samples[index][1],
        }
        return data_torch

    def __len__(self):
        return len(self.sampels)


class TestingCharadesDataset(object):
    def __init__(self, img_dir, csv_path, batch_size):
        def __init__(self):
            f = open('../data/vocab.json')
            self.vocab = json.loads(f.read())[0]
            self.feature_path = "../data/Charades_v1_features_rgb"
            self.unit_size = 1
            self.feature_dim = 4096
            self.pkl_path = "../data/Charades_feature_rgb_pkl"
            self.window_size = 192
            self.window_step = 64
            self.num_classes = 2  # action categroies + BG for THUMOS14 is 21
            self.mode = 'Train'
            self.sampels = []
            if self.mode == 'Train':
                self.clip_gt_path = "../data/charades_sta_train.txt"
            elif self.mode == 'Val':
                self.clip_gt_path = "../data/charades_sta_train.txt"
                self.window_step = 64
            elif self.mode == 'Valtrain':
                self.clip_gt_path = "../data/charades_sta_train.txt"
                self.window_step = 64
            elif self.mode == 'Test':
                self.clip_gt_path = "../data/charades_sta_test.txt"
                self.window_step = 64
            # multi thread
            self.exit_flag = False
            self.queue_lock = threading.Lock()
            self.queue = queue.Queue()
            self.item_done = 0
            self.movie_clip_featmaps_map = {}
            self.movie_clip_sentences_map = {}
            self.movie_length_info = {}
            self.movie_names_set = set()
            self._preparedata()
            print(
                'The number of {} dataset samples is {}'.format(self.mode, len(self.sampels)))

    def _preparedata(self):

        print('wait...prepare data')
        window_size = self.window_size
        stride = self.window_step
        id = 0

        # multithread
        for _ in range(8):
            thread = threading.Thread(target=self.process_thread, args=[self])
            thread.start()
        last_name = ''
        with open(self.clip_gt_path) as f:
            self.queue_lock.acquire()
            l_list = []
            for l in f:
                video_name = l.rstrip().split(" ")[0]
                if video_name == last_name or last_name == '':
                    l_list.append(l)
                else:
                    if len(l_list):
                        self.queue.put(l_list)
                    l_list = [l]
                last_name = video_name
                id = id + 1
            if len(l_list):
                self.queue.put(l_list)
            self.queue_lock.release()

        while not self.queue.empty():
            last = self.item_done
        #            time.sleep(10)
        # if self.item_done % 100 == 0:
        #     print(self.item_done, (self.item_done - last) / 10)
        # if self.item_done > 10 and self.mode=='Val':
        #    break
        # if self.item_done > 100:
        #    break

        self.exit_flag = True


    def _get_feature(self, video_name, start, end):
        feat_dir = self.feature_path
        swin_step = 4
        all_feat = np.zeros([0, self.feature_dim], dtype=np.float32)
        current_pos = start * 4
        end = (end - 1) * 4
        feat_path = os.path.join(feat_dir, video_name)

        pkl_path = self.pkl_path
        # if os.path.exists(pkl_path) == False:
        #     os.mkdir(pkl_path)
        pickle_path = os.path.join(pkl_path, video_name + '_f.pkl')

        if os.path.exists(pickle_path):
            p_f = open(pickle_path, 'rb')
            all_feat = pickle.load(p_f)
            p_f.close()
            return all_feat
        while current_pos < end:
            clip_name = video_name + '-' + str(current_pos + 1).zfill(6) + '.txt'
            path = os.path.join(feat_path, clip_name)
            if os.path.exists(path):
                feat = np.loadtxt(path)
            else:
                # print(video_name,current_pos + 1)
                feat = np.zeros(self.feature_dim, dtype=np.float32)
            all_feat = np.vstack((all_feat, feat))
            current_pos += swin_step

        # print(all_feat.shape)
        # time.sleep(100)

        p_f = open(pickle_path, 'wb')
        pickle.dump(all_feat, p_f)
        p_f.close()
        return all_feat  # window_size*4096

    def process_thread(sel, self):
        while not self.exit_flag:
            self.queue_lock.acquire()
            if not self.queue.empty():
                l = self.queue.get()
                self.item_done += 1
                self.queue_lock.release()
                self.process_video(l)
            else:
                self.queue_lock.release()

    def process_video(self, l_list):
        window_size, stride = self.window_size, self.window_step
        l_length = len(l_list)
        l = l_list[0]
        video_name = l.rstrip().split(" ")[0]
        clip_list = os.listdir(self.feature_path + '/' + video_name)
        length = len(clip_list)  # temporal length]
        video_feature_total = self._get_feature(video_name, 0, length)
        video_feature_total = torch.Tensor(video_feature_total)

        for id in range(l_length):
            l = l_list[id]
            gt_start = float(l.rstrip().split(" ")[1])
            gt_end = float(l.rstrip().split(" ")[2].split("##")[0])
            sentence = l.rstrip().split("##")[1][:-1]

            if gt_start > gt_end:
                gt_start, gt_end = gt_end, gt_start

            round_gt_start = round(gt_start * 24 / 4) * 4
            round_gt_end = round(gt_end * 24 / 4) * 4
            n_window = math.ceil((length + stride - window_size) / stride)
            # print(n_window, length)
            frameList = [1 + self.unit_size * i for i in range(length)]  # 1, 2, 3
            windows_start = [i * stride for i in range(int(n_window))]

            word_indexes = []
            for word in sentence.split(' '):
                idx = self.vocab.get(word)
                if idx is not None:
                    word_indexes.append(idx)
                else:
                    word_indexes.append(0)  # <UNK>
            sent = np.array(word_indexes, dtype=np.long)
            flag = 1
            if length < window_size:
                windows_start = [0]
                frameList.extend([frameList[-1] + self.unit_size * (i + 1) for i in range(window_size - length)])

            elif length - windows_start[-1] - window_size > 0:
                flag = 0
                windows_start.append(length - window_size)
            xmin = [(i - 1) * 4 + 1 for i in frameList]
            xmax = frameList[1:]
            xmax.append(frameList[-1] + 1)
            xmax = [(i - 1) * 4 + 1 for i in xmax]

            for start in windows_start:
                # tmp_data = _get_feature(self.feature_path, video_name, start, start+window_size)
                window = min(window_size, video_feature_total.shape[0] - start)
                pad_dim = window_size + start - video_feature_total.shape[0]
                tmp_data = video_feature_total[start:start + window, :]  # [window_size, feature_dim]
                if pad_dim > 0:
                    pad = torch.zeros(pad_dim, self.feature_dim)
                    tmp_data = torch.cat((tmp_data, pad), 0)
                self.movie_clip_featmaps_map[video_name].append(
                    [video_name + '%' + str(start) + '%' + str(start + window), tmp_data])
                # if(tmp_data.shape[0] != 128):
                #    print(video_name, sentence,start, start + window_size,video_feature_total.shape)
                #    time.sleep(1000000)
                tmp_anchor_xmins = xmin[start:start + window_size]
                tmp_anchor_xmaxs = xmax[start:start + window_size]
                # gt_box -iou;   label:0/1;    feat; query
                tmp_ioa = ioa_with_anchors(round_gt_start, round_gt_end, tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1])
                if tmp_ioa > 0:
                    # gt bbox info
                    corrected_start = max(round_gt_start, tmp_anchor_xmins[0]) - tmp_anchor_xmins[0]
                    corrected_end = min(round_gt_end, tmp_anchor_xmaxs[-1]) - tmp_anchor_xmins[0]

                    corrected_start = max(corrected_start, 0.0)
                    corrected_start = min(corrected_start, self.window_size * 4)
                    corrected_end = max(corrected_end, 0.0)
                    corrected_end = min(corrected_end, self.window_size * 4)

                    tmp_gt_bbox = [int(float(corrected_start) / 4),
                                   int(float(corrected_end) / 4)]
                else:
                    continue
                label = 0
                if (tmp_ioa > 0 and flag == 0 and tmp_anchor_xmins[0] == xmin[0]):
                    label = 1
                ###Todo add sample balance : label = 1 positive; label = 0 negative

                #                        print(video_name, round_gt_start, round_gt_end, tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1],tmp_gt_bbox, label,  tmp_data.shape, tmp_ioa)
                # the overlap region is corrected
                tmp_results = [tmp_data, np.array(tmp_gt_bbox), np.array(sent),
                               np.array([round_gt_start, round_gt_end])]
                if self.mode != 'Train':
                    tmp_results.append(video_name)
                    tmp_results.append(sentence)
                    tmp_results.append(tmp_anchor_xmins[0])
                    if video_name not in self.movie_length_info.keys():
                        self.movie_length_info[video_name] = length
                    self.movie_clip_sentences_map[video_name].append(
                        [video_name + '%' + str(tmp_gt_bbox[0]) + '%' + str(tmp_gt_bbox[1]), sent])
                # print(video_name, sentence, tmp_data.shape, tmp_ioa, [round_gt_start, round_gt_end],[tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1]],tmp_gt_bbox)
                # time.sleep(10)
                self.sampels.append(tmp_results)
                self.movie_names = list(self.movie_names_set)


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
        self.window_size = 100#768#config.window_size

        self.mode = 'Train'
        if self.mode == 'Train':
            self.window_step = 100#128#config.window_step
        else:
            self.window_step = 100#256#config.inference_window_step

        self.feature_dim = 500#config.feature_dim
        self.samples = []

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

        print('The number of {} dataset Activitynet samples is {}'.format(self.mode, len(self.samples)))

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
            movie_name = clip_name.split("_")[0]   #such as, s32-d55.avi
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
        return len(self.samples)


    def sent_convert_to_vec(self, sent):
        cfg.ntoken, cfg.word_dim


    def __getitem__(self, index):

        s = self.samples[index]
        
        
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
        sent_raw = self.samples[index][2]

        #bs = list(self.samples[index][0].size())[0]
        data_torch = {
            'vis': self.samples[index][0],#.reshape(bs, -1),
            'sent': sent_raw,
            'offset': self.samples[index][1],
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
        for video_name in list(self.data.keys()):
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
                        #sent 为word的index集合，sentences为实际单词
                        tmp_results = [tmp_data, np.array(tmp_gt_bbox),
                                       np.array(sent), np.array([round_gt_start, round_gt_end])]

                        if self.mode != 'Train':
                            # print(xmin, tmp_gt_bbox, round_gt_start, round_gt_end)
                            tmp_results.append(video_name)
                            tmp_results.append(sentences[idx])
                            tmp_results.append(xmin)
                            tmp_results.append(ratio)
                        self.samples.append(tmp_results)

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
        self.window_size = 100#768  # config.window_size

        self.mode = 'Test'
        if self.mode == 'Train':
            self.window_step = 100#128  # config.window_step
        else:
            self.window_step = 100#256  # config.inference_window_step

        self.feature_dim = 500  # config.feature_dim
        self.samples = []

        if self.mode == 'Train':
            self.data = read_json('../data/train.json')
        elif self.mode == 'Val':
            self.data = read_json('../data/val.json')

        elif self.mode == 'Valtrain':
            self.data = read_json('../data/val.json')

        elif self.mode == 'Test':
            self.data = read_json('../data/test.json')

        self.features_h5py = tables.open_file('../data/sub_activitynet_v1-3.c3d.hdf5', 'r')
        self.movie_clip_featmaps_map = {}
        self.movie_clip_sentences_map = {}
        self.movie_length_info = {}
        self.movie_names_set = set()
        self._process_video()
        print('The number of {} dataset Activitynet samples is {}'.format(self.mode, len(self.samples)))


        self.batch_size = batch_size


    def _process_video(self):  # 6*feature
        count = 0

        for video_name in self.data.keys():
            if video_name not in self.movie_clip_featmaps_map.keys():
                self.movie_clip_featmaps_map[video_name] = []
            if video_name not in self.movie_clip_sentences_map.keys():
                self.movie_clip_sentences_map[video_name] = []
            count += 1
            if count > 100 and self.mode == 'Val':
                break
            if count > 30 and self.mode == 'Test':
                break
            d = self.data[video_name]
            sentences = d['sentences']
            # data = torch.Tensor(self.features[video_name])
            data = torch.Tensor(self.features_h5py.root[video_name]['c3d_features'].read())
            times = d['timestamps']  #时间最大值
            length = data.shape[0]
            ratio = float(length) / float(d['duration'])  # ~3.7,3.8  每秒对应的维度 124.23对于[463,500]
            length = int(d['duration'] * ratio)      #总维度

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
                self.movie_clip_featmaps_map[video_name].append(
                    [video_name + '%' + str(start) + '%' + str(start + window), tmp_data])
                for idx in range(len(pair)):
                    sent = pair[idx][0]
                    round_gt_start = pair[idx][1][0]
                    round_gt_end = pair[idx][1][1]

                    tmp_ioa = ioa_with_anchors(round_gt_start, round_gt_end, xmin, xmax)

                    # 有gt的片段被送去训练 test时也只送有交集的框 todo fix 另外测试无的部分
                    if tmp_ioa > 0:
                        # gt bbox info
                        corrected_start = max(round_gt_start, xmin) - xmin     #相对坐标
                        corrected_end = min(round_gt_end, xmax) - xmin         #相对坐标
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
                        
                        #print(np.array(sent).shape) #(10,) torch.Size([768, 500])
                        #print(tmp_data.shape)
                        tmp_results = [tmp_data, np.array(tmp_gt_bbox),
                                       np.array(sent), np.array([round_gt_start, round_gt_end])]

                        if self.mode != 'Train':
                            tmp_results.append(video_name)
                            tmp_results.append(sentences[idx]) # d['sentences']
                            tmp_results.append(xmin)  #window_start
                            tmp_results.append(ratio)
                            self.movie_names_set.add(video_name)
                            if video_name not in self.movie_length_info.keys():
                                self.movie_length_info[video_name] = length
                            self.movie_clip_sentences_map[video_name].append([video_name + '%' + str(tmp_gt_bbox[0]) + '%' + str(tmp_gt_bbox[1]),sent])
                        self.samples.append(tmp_results)
                        self.movie_names = list(self.movie_names_set)
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
                self.clip_sentence_pairs.append((clip_name, sent_vec)) #such as(s32-d55.avi_169_324 ,(13, 4800))
        print(str(len(self.clip_sentence_pairs))+" pairs are readed")
        movie_names_set = set()
        self.movie_clip_names = {}
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]  #s13-d21.avi_252_452
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)    #存入该视频对应的所有clip的索引
        self.movie_names = list(movie_names_set)
        self.clip_num_per_movie_max = 0
        for movie_name in self.movie_clip_names:    #限制一个视频被clip的个数
            if len(self.movie_clip_names[movie_name])>self.clip_num_per_movie_max: self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])
        print("Max number of clips in a movie is "+str(self.clip_num_per_movie_max))
        
        self.sliding_clip_path = img_dir # "./exp_data/Interval128_256_overlap0.8_c3d_fc6/"
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.sliding_clip_names = []
        for clip_name in sliding_clips_tmp:     #such as,s37-d46.avi_9997_10509.npy
            if clip_name.split(".")[2]=="npy":
                movie_name = clip_name.split("_")[0] #such as,s37-d46.avi
                if movie_name in self.movie_clip_names:    #如果这个sliding视频有clip过的词向量
                    self.sliding_clip_names.append(clip_name.split(".")[0]+"."+clip_name.split(".")[1])# such as,s37-d46.avi_9997_10509
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
            if movie_name in self.clip_sentence_pairs[k][0]: #such as, s32-d55 in (s32-d55.avi_169_324 ,(13, 4800))[0]
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

