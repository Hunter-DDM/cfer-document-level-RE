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

# load
def load_data(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        line = lines[0]
        dicts = json.loads(line)

    return dicts


def get_relation(train_data, dev_data):
    relation = []
    for d in train_data:
        labels = d['labels']
        for label in labels:
            r = label['r']
            relation.append(r)
    for d in dev_data:
        labels = d['labels']
        for label in labels:
            r = label['r']
            relation.append(r)
    relation = list(set(relation))
    relation = sorted(relation)

    for i, r in enumerate(relation):
        config.relation2id[r] = i
        config.id2relation[i] = r

    config.relation2id['None'] = len(relation)
    config.id2relation[len(relation)] = 'None'


def data_process(data):
    """preprocessing"""
    for document in data:
        title = document['title']
        sents = document['sents']
        vertexSet = document['vertexSet']
        labels = document['labels']

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

        triples = []
        for label in labels:
            h = label['h']
            t = label['t']
            r = label['r']
            r_id = config.relation2id[r]
            triples.append((h, r_id, t))

        document['sent_idx'] = sent_idx
        document['word_num'] = word_num
        document['triples'] = triples

    return data


""" **Data Loader** """


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, data, evaluation=False):
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
                    sent_ids = sent_ids[:len(sent_ids) - len(id)]
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
        commons = [0 for _ in range(word_num)]

        for i, mentions in enumerate(vertexSet):
            for mention in mentions:
                global_pos = mention['global_pos']
                for p in global_pos:
                    commons[p] = i + 1

        return commons

    def get_target(self, triples, vertexSet):
        entity_num = len(vertexSet)
        target = np.zeros((entity_num, entity_num, config.R), dtype=int)
        target[:, :, -1] = 1

        for triple in triples:  # triple: (h, r, t)
            target[triple[0], triple[2], triple[1]] = 1
            target[triple[0], triple[2], -1] = 0

        return target

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
        triples = document['triples']

        sent_ids, global_spans, masks = self.get_ids(sents)
        distances = self.get_distances(vertexSet)
        commons = self.get_commons(vertexSet, word_num)
        types = self.get_types(vertexSet, global_spans)
        target = self.get_target(triples, vertexSet)

        return document, types, target, sent_ids, global_spans, masks, distances, commons
