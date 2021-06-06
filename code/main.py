import argparse
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
from data_loader import *
from model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=config.log_path,
                    filemode='a')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)


def loss_fn(pred, target):
    """
    pred: (n*n, R)
    target: (n*n, R)
    """
    loss_fct = nn.BCEWithLogitsLoss(reduction='mean')

    return loss_fct(pred, target)


def train_fn(data_loader, data_adjs, data_paths, model, optimizer, scheduler, ema):
    model.train()
    all_loss = 0.
    data_index = [i for i in range(len(data_loader))]
    random.shuffle(data_index)
    tk0 = tqdm(data_index, total=len(data_index), desc="Training")

    model.zero_grad()

    for bi, index in enumerate(tk0):
        document, types, target, sents_ids, global_spans, masks, all_distances, commons = data_loader[index]
        adj = data_adjs[index]
        paths_list = data_paths[index]  # (entity_num*entity_num, 70)
        vertexSet = document['vertexSet']
        n = len(vertexSet)

        adj = torch.tensor(adj, dtype=torch.float)
        types = torch.tensor(types, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.float).view(-1, config.R)  # (entity_num*entity_num, 97)
        sents_ids = torch.tensor(sents_ids, dtype=torch.long)  # (1 or 2, id_num)

        masks = torch.tensor(masks, dtype=torch.long)
        commons = torch.tensor(commons, dtype=torch.long)

        post_count = int(target[:, :-1].sum().item())
        idx = [i for i in range(target.shape[0])]
        indices = []
        mask_count = 0
        random.shuffle(idx)
        if post_count == 0:
            for i in idx[:20]:
                indices.append(i)
        else:
            for i in idx:
                if len(indices) >= 173:
                    break
                if target[i][-1].item() == 0:
                    indices.append(i)
                if target[i][-1].item() == 1 and mask_count < post_count * config.sample_rate:
                    indices.append(i)
                    mask_count += 1
        random.shuffle(indices)

        paths_indexs = [paths_list[i] for i in indices]
        distances = [all_distances[i] for i in indices]

        head_entity_index = []
        tail_entity_index = []

        for i in indices:
            head_index = i // n
            tail_index = i % n
            head_entity_index.append(head_index)
            tail_entity_index.append(tail_index)

        sents_ids = sents_ids.to(config.device)

        types = types.to(config.device)
        adj = adj.to(config.device)
        masks = masks.to(config.device)
        commons = commons.to(config.device)
        indices = torch.tensor(indices, dtype=torch.long)
        target = torch.index_select(target, 0, indices).to(config.device)

        pred = model(sents_ids, masks, vertexSet, types, adj, global_spans, paths_indexs, head_entity_index,
                     tail_entity_index, commons, distances)

        loss = loss_fn(pred, target)
        tk0.set_postfix(loss=loss.item())
        loss = loss / config.batch_size
        loss.backward()

        if (bi > 0 and bi % config.batch_size == 0) or bi == len(data_index) - 1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # EMA
            ema(model, config.num_updates)
            config.num_updates += 1

        all_loss += loss.item()

    return all_loss / len(data_loader)


def eval(data_loader, data_adjs, data_paths, model):
    model.eval()

    pred_counts = [0 for _ in range(len(config.thresholds))]
    pred_rights = [0 for _ in range(len(config.thresholds))]
    right = 0

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
        for bi, data in enumerate(tk0):
            document, types, target, sents_ids, global_spans, masks, all_distances, commons = data_loader[bi]
            adj = data_adjs[bi]
            paths_indexs = data_paths[bi]

            vertexSet = document['vertexSet']
            triples = document['triples']

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
                         tail_entity_index, commons, all_distances)

            pred = pred.view(len(vertexSet), len(vertexSet), -1)  # (mention_num, mention_num, 97)
            pred = pred[:, :, :-1]
            pred = F.sigmoid(pred)

            right += len(triples)

            for i, threshold in enumerate(config.thresholds):
                pred_ = (pred > threshold).nonzero().cpu().numpy()
                pred_counts[i] += pred_.shape[0]
                for j in range(pred_.shape[0]):
                    head = pred_[j][0]
                    tail = pred_[j][1]
                    relation = pred_[j][2]
                    if (head, relation, tail) in triples:
                        pred_rights[i] += 1

    best_P = 0.
    best_R = 0.
    best_F1 = 0.
    best_threshold = 0.
    for i, threshold in enumerate(config.thresholds):
        pred_count = pred_counts[i]
        pred_right = pred_rights[i]
        if pred_count == 0:
            P = 0
        else:
            P = pred_right / pred_count
        R = pred_right / right
        F1 = 0.
        if P + R > 0:
            F1 = 2 * P * R / (P + R)
        if best_F1 < F1:
            best_P = P
            best_R = R
            best_F1 = F1
            best_threshold = threshold

    return best_P, best_R, best_F1, best_threshold


def run(model, train_data_loader, dev_data_loader, train_adjs, dev_adjs, train_paths, dev_paths, optimizer, scheduler,
        ema):
    best_f1 = 0.
    best_threshold = 0.

    for epoch in range(config.EPOCHS):
        print('epoch:', epoch)
        loss = train_fn(
            train_data_loader,
            train_adjs,
            train_paths,
            model,
            optimizer,
            scheduler,
            ema
        )

        if (epoch < 100 and epoch % 10 == 0) or (epoch > 200 and epoch < 250) or (epoch > 250 and epoch % 5 == 0):
            # EMA assign
            ema.assign(model)

            P, R, F1, threshold = eval(
                dev_data_loader,
                dev_adjs,
                dev_paths,
                model
            )

            # EMA resume
            ema.resume(model)

            print('Dev: P:{}, R:{}, F1:{}, threshold:{}'.format(str(P)[:6], str(R)[:6], str(F1)[:6], threshold))
            logging.info('Dev: P:{}, R:{}, F1:{}, threshold:{}'.format(str(P)[:6], str(R)[:6], str(F1)[:6], threshold))

            if F1 > best_f1:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                model_path = config.save_path + "_{}_{}.pkl".format(threshold, str(F1)[:6])
                torch.save(checkpoint, model_path)
                best_f1 = F1
                best_threshold = threshold

                # save EMA
                with open(config.ema_path + str(config.num_updates) + '_{}.json'.format(threshold, str(F1)[:6]),
                          'wb') as json_file:
                    pickle.dump(ema.shadow, json_file)
                    print('ema saved!')

            print('best f1:', best_f1, 'best_threshold:', best_threshold)
            logging.info('best f1:{}, best_threshold:{}'.format(str(best_f1), best_threshold))


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



def main():
    train_data, dev_data, test_data, train_paths, dev_paths, train_adjs, dev_adjs = load_datas()

    train_data_loader = DataLoader(train_data)
    dev_data_loader = DataLoader(dev_data)

    model = Model().to(config.device)

    bert_no_decay = [p for n, p in model.named_parameters() if 'bert' in n and ('bias' in n or 'LayerNorm' in n)]
    bert_decay = [p for n, p in model.named_parameters() if 'bert' in n and 'bias' not in n and 'LayerNorm' not in n]
    other_params_decay = [p for n, p in model.named_parameters() if 'bert' not in n and 'bias' not in n]
    other_params_no_decay = [p for n, p in model.named_parameters() if 'bert' not in n and 'bias' in n]

    optimizer = AdamW([{'params': other_params_decay, 'lr': config.lr_other, 'weight_decay': 0.0001},
                       {'params': other_params_no_decay, 'lr': config.lr_other, 'weight_decay': 0},
                       {'params': bert_decay, 'lr': config.lr_bert, 'weight_decay': 0.0001},
                       {'params': bert_no_decay, 'lr': config.lr_bert, 'weight_decay': 0}]
                      )

    num_training_steps = config.EPOCHS * math.ceil(len(train_data_loader) // config.batch_size)
    num_warmup_steps = config.warmup_steps * num_training_steps

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    ema = EMA(mu=0.9999)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param)

    run(model, train_data_loader, dev_data_loader, train_adjs, dev_adjs, train_paths, dev_paths, optimizer, scheduler,
        ema)

main()
