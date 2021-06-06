import torch
from transformers import BertTokenizer

class config:
    # dataset
    train_data_path = './data/train_annotated.json'
    dev_data_path = './data/dev.json'
    test_data_path = './data/test.json'
    # path
    train_paths_path = './data/path/train_paths.npy'
    dev_paths_path = './data/path/dev_paths.npy'
    test_paths_path = './data/path/test_paths.npy'
    # adj matrix
    train_adjs_path = './data/adj/train_adjs.npy'
    dev_adjs_path = './data/adj/dev_adjs.npy'

    # save path
    save_path = './model/model_bert_512_para_ema2_test'
    ema_path = './model/model_bert_512_para_ema2_test'
    # log path
    log_path = './log/model_bert_512_para_ema2.log'
    # bert path
    bert_path = 'bert-base-uncased'
    len_head_entity_index = 0

    # hyper-parameters (to be tuned)
    device = torch.device('cuda:0')
    EPOCHS = 300
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    lr_other = 5e-4
    hidden_size = 512
    dropout_embedding = 0.2
    dropout_gru = 0.2
    dropout_attention = 0.4
    dropout_gcn = 0.6
    sample_rate = 3
    max_path_len = 55
    dis_num = 50
    common_num = 50
    batch_size = 32
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    lr_bert = 1e-5
    warmup_steps = 0.06
    num_updates = 0
    gcn_layer_first = 4
    gcn_layer_second = 4

    # hyper-parameters (do not tune)
    bert_size = 768
    common_emb_dim = 20
    dis_emb_dim = 20
    type_emb_dim = 100
    R = 97
    type2id = {'ORG': 1, 'TIME': 2, 'MISC': 3, 'LOC': 4, 'PER': 5, 'NUM': 6, 'None': 7}
    type_num = 7
    VOCAB_SIZE = 65507
    relation2id = {}
    id2relation = {}


