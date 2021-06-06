import copy
import math
import json
import time
import re
import random
import gc
import sys
import os
import logging
import pickle
from tqdm.autonotebook import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import BertTokenizer, BertModel, AdamW, BertConfig, BertPreTrainedModel, get_linear_schedule_with_warmup

from config import config

class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}  # cpu

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        decay = min(self.mu, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone().cpu()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name].cuda()

        self.original = {}


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, d, layers, dropout_gcn=config.dropout_gcn, self_loop=False):
        super(GraphConvLayer, self).__init__()
        self.d = d
        self.layers = layers
        self.head_dim = self.d // self.layers
        self.gcn_drop = nn.Dropout(dropout_gcn)

        self.linear_output = nn.Linear(self.d, self.d)

        self.linear_gate = nn.Linear(self.d, self.d)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.d + self.head_dim * i), self.head_dim))

        self.self_loop = self_loop

    def forward(self, adj, gcn_inputs):
        '''
        adj: (n, n)
        gcn_inputs: (n, d)
        '''
        denom = adj.sum(-1).unsqueeze(-1) + 1  # (n, 1)

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = torch.matmul(adj, outputs)  # (n, n) * (n, d) -> (n, d)
            AxW = self.weight_list[l](Ax)
            if self.self_loop:
                AxW = AxW + self.weight_list[l](outputs)  # self loop
            else:
                AxW = AxW

            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=1)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=1)
        gcn_outputs = gcn_outputs + gcn_inputs

        # high way
        gate = torch.sigmoid(self.linear_gate(gcn_outputs))
        out = self.linear_output(gcn_outputs)
        # drop=0.1 + relu
        out = gate * out + (1 - gate) * gcn_outputs

        return out


class Attention(nn.Module):
    def __init__(self, d, dis_emb_dim):
        super(Attention, self).__init__()
        self.d = d
        self.dis_emb_dim = dis_emb_dim
        self.linear = nn.Linear(self.d * 4 + self.dis_emb_dim, 1)

    def forward(self, entity, path, dis_embs):
        """
        entity, path: (k, d*2)
        dis_embs: (k, dis_emb_dim)
        """
        inputs = torch.cat((entity, path, dis_embs), 1)  # (self.d*4, 1)

        scores = F.softmax(self.linear(inputs), 0)  # (k, 1)

        return scores


class Model(nn.Module):
    def __init__(self, weight_matrix=None):
        super(Model, self).__init__()
        self.bert_size = config.bert_size
        self.hidden_size = config.hidden_size
        self.type_emb_dim = config.type_emb_dim
        self.common_emb_dim = config.common_emb_dim
        self.dis_emb_dim = config.dis_emb_dim
        self.max_path_len = config.max_path_len
        self.gcn_layer_first = config.gcn_layer_first
        self.gcn_layer_second = config.gcn_layer_second

        # encoder
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.type_embedding = nn.Embedding(config.type_num, self.type_emb_dim)
        self.com_embedding = nn.Embedding(config.common_num, self.common_emb_dim)
        self.dropout_embedding = nn.Dropout(config.dropout_embedding)

        # GRU
        self.gru = nn.GRU(self.bert_size + self.type_emb_dim + self.common_emb_dim, self.hidden_size // 2,
                          batch_first=True, bidirectional=True)
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        self.dropout_gru = nn.Dropout(config.dropout_gru)

        # dcgcn
        self.gcns = nn.ModuleList()
        self.gcns.append(GraphConvLayer(self.hidden_size, self.gcn_layer_first))
        self.gcns.append(GraphConvLayer(self.hidden_size, self.gcn_layer_second))

        # path + attention
        self.gru2 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        for name, param in self.gru2.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        self.dis_embedding = nn.Embedding(config.dis_num, self.dis_emb_dim)
        self.attention = Attention(self.hidden_size, self.dis_emb_dim)
        self.dropout_attention = nn.Dropout(config.dropout_attention)

        # classification
        self.bilinear = nn.Bilinear(self.hidden_size * 2, self.hidden_size * 2, config.R)

    def get_word_embs(self, words_embs, global_spans):
        res_embs = torch.zeros((len(global_spans), self.bert_size)).to(config.device)
        for i, span in enumerate(global_spans):
            start = span[0]
            end = span[-1]
            res_embs[i, :] = torch.mean(words_embs[start:end + 1], 0, True)

        return res_embs

    def get_entity_embs(self, words_embs, vertexSet):
        entity_embs = torch.zeros((len(vertexSet), self.hidden_size)).to(config.device)
        for i, mentions in enumerate(vertexSet):
            cur_mention_embs = torch.zeros((len(mentions), self.hidden_size)).to(config.device)
            for j, mention in enumerate(mentions):
                global_pos = mention['global_pos']
                indices = torch.tensor(global_pos, dtype=torch.long).to(config.device)
                cur_mention_embs[j, :] = torch.mean(torch.index_select(words_embs, 0, indices), 0, True)
            entity_embs[i, :] = torch.mean(cur_mention_embs, 0, True)

        return entity_embs

    def get_ht_embs(self, words_embs, paths_indexs, head_entity_embs, tail_entity_embs, distances):
        head_embs = []
        tail_embs = []
        for i, paths_index in enumerate(paths_indexs):  # search all paths (m, *path_num, max_len)
            k = len(paths_index)
            path_pad_u = torch.zeros((len(paths_index), self.max_path_len, self.hidden_size)).to(config.device)
            lengths = []
            for j, path_index in enumerate(paths_index):
                indices = torch.tensor(path_index, dtype=torch.long).to(config.device)
                path_pad_u[j, :len(path_index), :] = torch.index_select(words_embs, 0, indices)
                lengths.append(len(path_index))
            lengths = torch.tensor(lengths).to(config.device)
            lens_sorted, lens_argsort = torch.sort(lengths, 0, True)
            lens_sorted = lens_sorted.to(config.device)
            lens_argsort = lens_argsort.to(config.device)

            path_pad_u = torch.index_select(path_pad_u, 0, lens_argsort)
            packed = pack_padded_sequence(path_pad_u, lens_sorted, batch_first=True)
            output, h = self.gru2(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
            h = h.permute(1, 0, 2)  # (k, 2, hidden_size)

            head_path_embs = h[:, 1, :]  # (k, hidden_size)
            tail_path_embs = h[:, 0, :]

            # Attention
            if k > 1:
                # distance embedding
                distance = torch.tensor(distances[i], dtype=torch.long).to(config.device)
                dis_embs = self.dis_embedding(distance)  # (k, dis_emb_dim)

                head_entity_emb = head_entity_embs[i].view(1, -1).repeat(k, 1)  # (1, hidden_size) -> (k, hidden_size)
                tail_entity_emb = tail_entity_embs[i].view(1, -1).repeat(k, 1)
                entity_cat = torch.cat((head_entity_emb, tail_entity_emb), 1)  # (k, hidden_size) -> (k, hidden_size*2)
                path_cat = torch.cat((head_path_embs, tail_path_embs), 1)
                path_weight = self.attention(entity_cat, path_cat, dis_embs)  # (k, 1)
                head_path_embs = path_weight * head_path_embs  # (k, hidden_size)
                tail_path_embs = path_weight * tail_path_embs

            head_emb = torch.sum(head_path_embs, 0, True)  # (1, hidden_size)
            tail_emb = torch.sum(tail_path_embs, 0, True)
            head_embs.append(head_emb)
            tail_embs.append(tail_emb)

        head_embs = torch.cat(head_embs).to(config.device)
        tail_embs = torch.cat(tail_embs).to(config.device)

        return head_embs, tail_embs

    def forward(self, sents_ids, masks, vertexSet, types, adj, global_spans, paths_indexs, head_entity_index,
                tail_entity_index, commons, distances):
        hs = self.bert(input_ids=sents_ids, attention_mask=masks)[0]  # (1 or 2, id_num, hidden_size)

        if sents_ids.shape[0] == 1:
            words_embs = hs.view(-1, hs.shape[-1])[1:-1]  # (id_num-2, hidden_size)
        else:
            words_embs1 = hs[0, 1:-1]
            indices = (masks[1] == True).nonzero().view(-1).to(config.device)
            words_embs2 = torch.index_select(hs[1], 0, indices)[1:-1]
            words_embs = torch.cat((words_embs1, words_embs2))

        words_embs = self.get_word_embs(words_embs, global_spans)  # (id_num, bert_size) -> (word_num, bert_size)
        assert words_embs.shape[0] == adj.shape[0]

        types_embs = self.type_embedding(types)  # (word_num) -> (word_num, type_emb_dim)
        common_embs = self.com_embedding(commons)  # (word_num) -> (word_num, common_emb_dim)
        words_embs = torch.cat((words_embs, types_embs, common_embs), 1)  # -> (word_num, bert_size+type_emb_dim)
        words_embs = self.dropout_embedding(words_embs)

        # global encoding
        words_embs = words_embs.view(1, words_embs.shape[0],
                                     words_embs.shape[1])  # -> (1, word_num, type_size+bert_size)
        words_embs = self.gru(words_embs)[0].squeeze()  # (word_num, hidden_size)
        words_embs = self.dropout_gru(words_embs)

        # GCN
        for i in range(len(self.gcns)):
            words_embs = self.gcns[i](adj, words_embs)

        # etity embs
        etity_embs = self.get_entity_embs(words_embs, vertexSet)  # (entity_num, hidden_size)

        # entity pair embs
        head_entity_indices = torch.tensor(head_entity_index, dtype=torch.long).to(config.device)
        head_entity_embs = torch.index_select(etity_embs, 0, head_entity_indices)
        tail_entity_indices = torch.tensor(tail_entity_index, dtype=torch.long).to(config.device)
        tail_entity_embs = torch.index_select(etity_embs, 0, tail_entity_indices)

        # entity pairs' head & tail embs
        head_embs, tail_embs = self.get_ht_embs(words_embs, paths_indexs, head_entity_embs, tail_entity_embs, distances)
        head = torch.cat((head_entity_embs, head_embs), 1)
        tail = torch.cat((tail_entity_embs, tail_embs), 1)
        head = self.dropout_attention(head)
        tail = self.dropout_attention(tail)

        pred = self.bilinear(head, tail)

        return pred  # (entity_pair_num, R)

