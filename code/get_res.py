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

import spacy
from spacy.tokens import Doc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import BertTokenizer, BertModel, AdamW, BertConfig, BertPreTrainedModel, \
    get_linear_schedule_with_warmup

# configurations for data

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from config import config
from data_loader import get_relation, load_data
from model import *

config.model_path = '/home/renjing/cfer/models/model_bert_512_para_ema2_0.8_0.6075.pkl'
config.ema_path = '/home/renjing/cfer/models/model_bert_512_para_ema219776_0.8.json'
config.res_save_path = 'result.json'

config.thresholds = [0.027206406, 0.7347192, 0.8, 0.8500842, 0.2866432, 0.8824334, 0.8, 0.07099054, 0.8, 0.33777493,
                  0.8675586, 0.14178617, 0.8, 0.0017886619, 0.018898232, 0.8, 0.8, 0.29970348, 0.8, 0.0028602716,
                  0.95426935, 0.8, 0.8, 0.1495586, 0.6003812, 0.049580034, 0.16661401, 0.77043945, 0.97031116,
                  0.9935579, 0.8, 0.9169548, 0.042989288, 0.8, 0.1299838, 0.9070824, 0.8, 0.7215598, 0.99140394,
                  0.54074734, 0.9154665, 0.78640866, 0.99458814, 0.8, 0.8, 0.39556777, 0.33989364, 0.7894069, 0.078295,
                  0.8, 0.1906913, 0.8, 0.02231309, 0.8, 0.49018496, 0.12782995, 0.20293304, 0.17651184, 0.8, 0.45584375,
                  0.74141777, 0.5691519, 0.74046874, 0.8812864, 0.8, 0.8, 0.8, 0.99002403, 0.6128277, 0.662927, 0.8,
                  0.8376514, 0.99243224, 0.9942542, 0.8792411, 0.8, 0.99344105, 0.7274136, 0.97489274, 0.8, 0.018518208,
                  0.99547833, 0.879309, 0.01869425, 0.8, 0.9983028, 0.8, 0.9657363, 0.8, 0.31774586, 0.8, 0.4790035,
                  0.8, 0.8, 0.8, 0.8]

config.threshold = 0.8

def data_process(data):
    """preprocessing"""
    for document in data:
        title = document['title']
        sents = document['sents']
        vertexSet = document['vertexSet']

        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                if '\xa0' in word or word == ' ':
                    sents[i][j] = "-"

        sent_idx = []
        start = 0
        word_num = 0
        for sent in sents:
            sent_idx.append(start)
            start += len(sent)
            word_num += len(sent)

        for mentions in vertexSet:
            for mention in mentions:
                pos = mention['pos']
                sent_id = mention['sent_id']
                global_pos = [pos[0] + sent_idx[sent_id], pos[1] + sent_idx[sent_id]]
                if len(global_pos) == 1:
                    mention['global_pos'] = global_pos
                else:
                    mention['global_pos'] = [i for i in range(global_pos[0], global_pos[-1])]

        document['sent_idx'] = sent_idx
        document['word_num'] = word_num

    return data

class MyTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words):
        return Doc(self.vocab, words=words)

class DataLoader_bert(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, data, evaluation=False):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.tokenizer = MyTokenizer(self.nlp.vocab)
        self.data = data

    def get_ids(self, sents):
        all_sent_ids = []

        global_spans = []
        sent_ids = []
        global_pos = 0

        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                id = config.tokenizer.encode(word, add_special_tokens=False)

                sent_ids += id
                cur_global_span = [i for i in range(global_pos, global_pos + len(id))]
                global_spans.append(cur_global_span)
                global_pos += len(id)

                if len(sent_ids) >= 510:
                    sent_ids = sent_ids[:len(sent_ids)-len(id)]
                    sent_ids = [101] + sent_ids + [102]
                    all_sent_ids.append(sent_ids)
                    sent_ids = []
                    sent_ids += id

        sent_ids = [101] + sent_ids + [102]
        all_sent_ids.append(sent_ids)

        if len(all_sent_ids) == 2:
            max_len = len(all_sent_ids[0])
            masks = np.zeros((2, max_len))
            masks[0, :] = 1.
            masks[1, :len(all_sent_ids[1])] = 1.
            all_sent_ids[1] = all_sent_ids[1] + [0] * (max_len - len(all_sent_ids[1]))  # padding
            assert len(all_sent_ids[0]) == len(all_sent_ids[1])
        else:
            max_len = len(all_sent_ids[0])
            masks = np.ones((1, max_len))

        return all_sent_ids, global_spans, masks

    def get_adj(self, sents, sent_idx, vertexSet, word_num):
        """
        sents: list of sentence
        sent_idx: 记录每个句子第一个词的全局index
        masks: Tensor of (sent_num, MAX_LEN)
        vertexSet: [{'name': 'Bergqvist', 'pos': [9, 10], 'sent_id': 5}, ...]
        spans: word span in ids
        word_num: 文档词总数
        """
        adj = np.zeros((word_num, word_num))

        root_idx = []
        for i, sent in enumerate(sents):
            doc = self.nlp(sent)
            for token in doc:
                cur_index = token.i + sent_idx[i]
                head_index = token.head.i + sent_idx[i]
                adj[cur_index][head_index] = 1
                adj[head_index][cur_index] = 1

                if token.dep_ == 'ROOT':
                    root_idx.append(cur_index)

        for i in range(len(root_idx) - 1):
            start = root_idx[i]
            end = root_idx[i + 1]
            adj[start][end] = 1
            adj[end][start] = 1

        for i in range(adj.shape[0]):
            adj[i][i] = 1

        for i in range(adj.shape[0] - 1):
            adj[i][i + 1] = 1
            adj[i + 1][i] = 1

        for mentions in vertexSet:
            mention_index = []
            for mention in mentions:
                global_pos = mention['global_pos']
                mention_index.append(global_pos[0])
            for i in mention_index:
                for j in mention_index:
                    adj[i][j] = 1
                    adj[j][i] = 1

        return adj

    def get_types(self, vertexSet, global_spans):
        types = np.zeros((len(global_spans)), dtype=int)
        for mentions in vertexSet:
            for mention in mentions:
                type_id = config.type2id[mention['type']]
                global_pos = mention['global_pos']
                for i in global_pos:
                    types[i] = type_id
        return types

    def get_distances(self, vertexSet):
        distances = []

        for i, head_mentions in enumerate(vertexSet):
            for j, tail_mentions in enumerate(vertexSet):
                if i == j:
                    distances.append([0])
                    continue
                distance = []
                for head_mention in head_mentions:
                    for tail_mention in tail_mentions:
                        head_sent_id = head_mention['sent_id']
                        tail_sent_id = tail_mention['sent_id']
                        distance.append(abs(head_sent_id - tail_sent_id))
                distances.append(distance)

        return distances

    def get_commons(self, vertexSet, word_num):
        commons = [0 for i in range(word_num)]

        for i, mentions in enumerate(vertexSet):
            for mention in mentions:
                global_pos = mention['global_pos']
                for p in global_pos:
                    commons[p] = i + 1

        return commons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return
            document: dict
            adj: np.array of (word_num, word_num)
            target: Tensor of (mention_num*mention_num, R)
            sents_ids: Tensor of (sent_num, max_seq_len)
            spans: list of span list
            masks: Tensor of (sent_num, max_seq_len)
        """
        if not isinstance(idx, int):
            raise TypeError
        if idx < 0 or idx >= len(self.data):
            raise IndexError

        document = self.data[idx]
        vertexSet = document['vertexSet']
        sents = document['sents']
        word_num = document['word_num']
        sent_idx = document['sent_idx']

        sent_ids, global_spans, masks = self.get_ids(sents)
        distances = self.get_distances(vertexSet)
        commons = self.get_commons(vertexSet, word_num)
        types = self.get_types(vertexSet, global_spans)
        adjs = self.get_adj(sents, sent_idx, vertexSet, word_num)

        return document, types, sent_ids, global_spans, masks, distances, commons, adjs


def get_res(data_loader, data_paths, model):
    model.eval()
    res = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
        for bi, data in enumerate(tk0):
            document, types, sents_ids, global_spans, masks, distances, commons, adj = data
            paths_indexs = data_paths[bi]  # (entity_num*entity_num, 70)
            vertexSet = document['vertexSet']
            title = document['title']
            adj = torch.tensor(adj, dtype=torch.float).to(config.device)
            types = torch.tensor(types, dtype=torch.long).to(config.device)
            sents_ids = torch.tensor(sents_ids, dtype=torch.long).to(config.device)
            masks = torch.tensor(masks, dtype=torch.long).to(config.device)
            commons = torch.tensor(commons, dtype=torch.long).to(config.device)

            indices = [i for i in range(len(paths_indexs))]
            head_entity_index = []
            tail_entity_index = []
            n = len(vertexSet)

            for i in indices:
                head_index = i // n
                tail_index = i % n
                head_entity_index.append(head_index)
                tail_entity_index.append(tail_index)

            model.zero_grad()
            pred = model(sents_ids, masks, vertexSet, types, adj, global_spans, paths_indexs, head_entity_index,
                         tail_entity_index, commons, distances)

            pred = F.sigmoid(pred)

            for i in range(pred.shape[0]):
                if pred[i, -1] > config.threshold:
                    pred[i, :] = 0.

            pred = pred[:, :-1]

            for i in range(pred.shape[0]):
                if i // n == i % n:
                    pred[i].fill_(0.)

            pred = pred.cpu().numpy()
            for i in range(96):
                threshold = config.thresholds[i]
                for j in range(pred.shape[0]):
                    score = pred[j][i]
                    if score >= threshold:
                        res.append({
                            "title": title,
                            "h_idx": j // n,
                            "t_idx": j % n,
                            "r": config.id2relation[i],
                            "evidence": []
                        })

    return res

def load_datas():
    train_data = load_data(config.train_data_path)
    dev_data = load_data(config.dev_data_path)
    test_data = load_data(config.test_data_path)

    get_relation(train_data, dev_data)

    train_data = data_process(train_data)
    dev_data = data_process(dev_data)

    train_paths = list(np.load(config.train_paths_path, allow_pickle=True))
    dev_paths = list(np.load(config.dev_paths_path, allow_pickle=True))

    train_adjs = np.load(config.train_adjs_path, allow_pickle=True)
    dev_adjs = np.load(config.dev_adjs_path, allow_pickle=True)

    train_adjs = list(train_adjs)
    dev_adjs = list(dev_adjs)

    return train_data, dev_data, test_data, train_paths, dev_paths, train_adjs, dev_adjs

model = Model().to(config.device)
checkpoint = torch.load(config.model_path)
model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
gc.collect()

with open(config.ema_path, 'rb') as f:
    shadow = pickle.load(f, encoding='bytes')
ema = EMA(mu=0.999)
ema.shadow = shadow


train_data = load_data(config.train_data_path)
dev_data = load_data(config.dev_data_path)
test_data = load_data(config.test_data_path)

get_relation(train_data, dev_data)


test_data = data_process(test_data)
test_paths = list(np.load(config.test_paths_path, allow_pickle=True))
test_data_loader = DataLoader_bert(test_data)

dev_data = data_process(dev_data)
dev_paths = list(np.load(config.dev_paths_path, allow_pickle=True))
dev_data_loader = DataLoader_bert(dev_data)

ema.assign(model)
res = get_res(test_data_loader, test_paths, model)
#res = get_res(dev_data_loader, dev_paths, model)

result = []
for r in res:
    r['h_idx'] = int(r['h_idx'])
    r['t_idx'] = int(r['t_idx'])
    result.append(r)

# save
with open(config.res_save_path, 'w') as f:
    json.dump(result, f)


