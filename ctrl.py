#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

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
        x = x.to('cuda')
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

class CTRL(nn.Module):
    def __init__(self, 
                visual_dim, 
                sentence_embed_dim,
                semantic_dim,
                middle_layer_dim,
                dropout_rate=0., 
                ):
        super(CTRL, self).__init__()
        self.semantic_dim = semantic_dim

        self.v2s_fc = nn.Linear(visual_dim, semantic_dim)
        self.s2s_fc = nn.Linear(sentence_embed_dim, semantic_dim)
        self.fc1 = nn.Conv2d(semantic_dim * 4, middle_layer_dim, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(middle_layer_dim, 3, kernel_size=1, stride=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.word_emb = WordEmbedding(9091, 300)#(config.ntoken, config.word_dim)
        self.word_emb.init_embedding('../data/pretrained_embedding_weights.npy')
        #self.sent_emb = SentenceEmbedding(config.word_dim, config.sent_hidden_dim, sent_layers, bidirect=sent_bidirect,dropout=0)
        self.sent_emb = SentenceEmbedding(300, 250, 1, bidirect=True, dropout=0)

    def forward(self, visual_feature, word_seqs):
        word_embs = self.word_emb(word_seqs)
        sentence_embed = self.sent_emb(word_embs)
        
        batch_size,_ = visual_feature.size()

        visual_feature = visual_feature.reshape(batch_size, -1)
        visual_feature = visual_feature.to('cuda')
        transformed_clip = self.v2s_fc(visual_feature)

        transformed_sentence = self.s2s_fc(sentence_embed)

        transformed_clip_norm = transformed_clip / transformed_clip.norm(2, dim=1, keepdim=True) # by row
        transformed_sentence_norm = transformed_sentence / transformed_sentence.norm(2, dim=1, keepdim=True) # by row

        # Cross modal combine: [mul, add, concat]
        vv_f = transformed_clip_norm.repeat(batch_size, 1).reshape(batch_size, batch_size, self.semantic_dim)
        ss_f = transformed_sentence_norm.repeat(1, batch_size).reshape(batch_size, batch_size, self.semantic_dim)
        mul_feature = vv_f * ss_f
        add_feature = vv_f + ss_f
        cat_feature = torch.cat((vv_f, ss_f), 2)
        cross_modal_vec = torch.cat((mul_feature, add_feature, cat_feature), 2)
        
        # vs_multilayer 
        out = cross_modal_vec.unsqueeze(0).permute(0,3,1,2)  # match conv op 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.permute(0,2,3,1).squeeze(0)

        return out

class CTRL_loss(nn.Module):
    def __init__(self, lambda_reg):
        super(CTRL_loss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, net, offset_label):
        batch_size = net.size()[0]
        sim_score_mat, p_reg_mat, l_reg_mat = net.split(1, dim=2)
        sim_score_mat = sim_score_mat.reshape(batch_size, batch_size)
        p_reg_mat = p_reg_mat.reshape(batch_size, batch_size).to("cuda")
        l_reg_mat = l_reg_mat.reshape(batch_size, batch_size).to("cuda")

        # make mask mat
        I_2 = 2.0 * torch.eye(batch_size)
        all1 = torch.ones([batch_size, batch_size])
        mask = all1 - I_2

        # loss cls, not considering iou
        I = torch.eye(batch_size).to("cuda")
        batch_para_mat = torch.ones([batch_size, batch_size]) / batch_size
        para_mat = (I + batch_para_mat.to("cuda")).to("cuda")
        
        mask = mask.to("cuda")
        sim_score_mat = sim_score_mat.to("cuda")
        all1 = all1.to("cuda")
        
        loss_mat = torch.log(all1 + torch.exp(torch.mul(mask, sim_score_mat))).to("cuda")
        loss_mat = torch.mul(loss_mat, para_mat)
        loss_align = torch.mean(loss_mat)

        # regression loss
        l_reg_diag = torch.mm(torch.mul(l_reg_mat, I), torch.ones([batch_size, 1]).to("cuda"))
        p_reg_diag = torch.mm(torch.mul(p_reg_mat, I), torch.ones([batch_size, 1]).to("cuda"))
        offset_pred = torch.cat((p_reg_diag, l_reg_diag), 1)

        loss_reg = torch.mean(torch.abs(offset_pred.float().to("cuda") - offset_label.float().to("cuda")))

        loss = loss_align + self.lambda_reg * loss_reg
        
        return  loss
                



