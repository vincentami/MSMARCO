from __future__ import print_function
import sys
import os
import os.path
import csv
import re
import random
import datetime
import numpy as np
import pandas as pd
import copy
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


class DataReader:
    def __init__(self, data_file, num_meta_cols, multi_pass):
        self.num_meta_cols = num_meta_cols
        self.multi_pass = multi_pass
        self.regex_drop_char = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space = re.compile('\s+')
        self.__load_vocab()
        self.__load_idfs()
        self.__init_data(data_file)
        self.__allocate_minibatch()
        self.none_idf_no = 0

    def __tokenize(self, s, max_terms):
        return self.regex_multi_space.sub(' ', self.regex_drop_char.sub(' ', s.lower())).strip().split()[:max_terms]

    def __load_vocab(self):
        global VOCAB_SIZE
        self.vocab = {}
        with open(DATA_FILE_VOCAB, mode='r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.vocab[row[0]] = int(row[1])
        VOCAB_SIZE = len(self.vocab) + 1

        print_message("vocab size:" + str(VOCAB_SIZE))

        none_index = 0
        embeddings = np.zeros((VOCAB_SIZE, NUM_HIDDEN_NODES), dtype=np.float32)
        with open(DATA_EMBEDDINGS, mode='r', encoding="utf-8") as f:
            for line in f:
                cols = line.split(",")
                idx = self.vocab.get(cols[0], 0)
                if idx > 0 and len(cols) == (NUM_HIDDEN_NODES + 1):
                    for i in range(NUM_HIDDEN_NODES):
                        embeddings[idx, i] = float(cols[i + 1])
                else:
                    if (idx == 0):
                        none_index = none_index + 1
                    else :
                        print_message("DATA_EMBEDDINGS idx:{} err:{},{}".format(idx, len(cols),line))

            print_message("DATA_EMBEDDINGS size:{} none_embeding:{}".format(embeddings.size, none_index))

        self.pre_trained_embeddings = torch.tensor(embeddings)

    def __load_idfs(self):
        self.idfs = {}
        with open(DATA_FILE_IDFS, mode='r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.idfs[row[0]] = float(row[1])
            print_message("idf size:" + str(len(self.idfs)))

    def __init_data(self, file_name):
        self.reader = open(file_name, mode='r', encoding="utf-8")
        self.num_docs = len(self.reader.readline().split('\t')) - self.num_meta_cols - 1
        self.reader.seek(0)

    def __allocate_minibatch(self):
        self.features = {}
        if ARCH_TYPE != 1:
            self.features['local'] = []
        if ARCH_TYPE > 0:
            self.features['dist_q'] = np.zeros((MB_SIZE, MAX_QUERY_TERMS), dtype=np.int64)
            self.features['mask_q'] = np.zeros((MB_SIZE, MAX_QUERY_TERMS, 1), dtype=np.float32)
            self.features['dist_d'] = []
            self.features['mask_d'] = []
        for i in range(self.num_docs):
            if ARCH_TYPE != 1:
                self.features['local'].append(np.zeros((MB_SIZE, MAX_DOC_TERMS, MAX_QUERY_TERMS), dtype=np.float32))
            if ARCH_TYPE > 0:
                self.features['dist_d'].append(np.zeros((MB_SIZE, MAX_DOC_TERMS), dtype=np.int64))
                self.features['mask_d'].append(np.zeros((MB_SIZE, MAX_DOC_TERMS, 1), dtype=np.float32))
        self.features['labels'] = np.zeros((MB_SIZE), dtype=np.int64)
        self.features['meta'] = []

    def __clear_minibatch(self):
        if ARCH_TYPE > 0:
            self.features['dist_q'].fill(np.int64(0))
            self.features['mask_q'].fill(np.float32(0))
        for i in range(self.num_docs):
            if ARCH_TYPE != 1:
                self.features['local'][i].fill(np.float32(0))
            if ARCH_TYPE > 0:
                self.features['dist_d'][i].fill(np.int64(0))
                self.features['mask_d'][i].fill(np.float32(0))
        self.features['meta'].clear()

    def get_minibatch(self):
        self.__clear_minibatch()
        for i in range(MB_SIZE):
            row = self.reader.readline()
            if row == '':
                if self.multi_pass:
                    self.reader.seek(0)
                    row = self.reader.readline()
                else:
                    break
            cols = row.split('\t')
            q = self.__tokenize(cols[self.num_meta_cols], MAX_QUERY_TERMS)
            ds = [self.__tokenize(cols[self.num_meta_cols + i + 1], MAX_DOC_TERMS) for i in range(self.num_docs)]
            if ARCH_TYPE != 1:
                for d in range(self.num_docs):
                    for j in range(len(ds[d])):
                        for k in range(len(q)):
                            if ds[d][j] == q[k]:
                                if q[k] in self.idfs:
                                    self.features['local'][d][i, j, k] = self.idfs[q[k]]
                                else:
                                    self.features['local'][d][i, j, k] = 0.0
                                    self.none_idf_no = self.none_idf_no + 1
            if ARCH_TYPE > 0:
                for j in range(self.num_docs + 1):
                    terms = q if j == 0 else ds[j - 1]
                    for t in range(len(terms)):
                        term = self.vocab.get(terms[t], 0)
                        if j == 0:
                            self.features['dist_q'][i, t] = term
                            self.features['mask_q'][i, t, 0] = 1
                        else:
                            self.features['dist_d'][j - 1][i, t] = term
                            self.features['mask_d'][j - 1][i, t, 0] = 1
            self.features['meta'].append(tuple(cols[:self.num_meta_cols]))
        return self.features

    def reset(self):
        self.reader.seek(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Duet(torch.nn.Module):
    def __init__(self, READER_TRAIN, device):
        super(Duet, self).__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, NUM_HIDDEN_NODES)
        self.embed.weight = nn.Parameter(READER_TRAIN.pre_trained_embeddings, requires_grad=True)
        self.duet_local = nn.Sequential(nn.Conv1d(MAX_DOC_TERMS, NUM_HIDDEN_NODES, kernel_size=1),
                                        nn.ReLU(),
                                        Flatten(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES * MAX_QUERY_TERMS, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE))
        self.duet_dist_q = nn.Sequential(nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=3),
                                         nn.ReLU(),
                                         nn.MaxPool1d(POOLING_KERNEL_WIDTH_QUERY),
                                         Flatten(),
                                         nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                         nn.ReLU()
                                         )
        self.duet_dist_d = nn.Sequential(nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=3),
                                         nn.ReLU(),
                                         nn.MaxPool1d(POOLING_KERNEL_WIDTH_DOC, stride=1),
                                         nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=1),
                                         nn.ReLU()
                                         )
        self.duet_dist = nn.Sequential(Flatten(),
                                       nn.Dropout(p=DROPOUT_RATE),
                                       nn.Linear(NUM_HIDDEN_NODES * NUM_POOLING_WINDOWS_DOC, NUM_HIDDEN_NODES),
                                       nn.ReLU(),
                                       nn.Dropout(p=DROPOUT_RATE),
                                       nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                       nn.ReLU(),
                                       nn.Dropout(p=DROPOUT_RATE))
        self.duet_comb = nn.Sequential(nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                       nn.ReLU(),
                                       nn.Dropout(p=DROPOUT_RATE),
                                       nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                       nn.ReLU(),
                                       nn.Dropout(p=DROPOUT_RATE),
                                       nn.Linear(NUM_HIDDEN_NODES, 1),
                                       nn.ReLU())
        self.scale = torch.tensor([0.1], requires_grad=False).to(device)

    def forward(self, x_local, x_dist_q, x_dist_d, x_mask_q, x_mask_d):
        if ARCH_TYPE != 1:
            h_local = self.duet_local(x_local)
        if ARCH_TYPE > 0:
            h_dist_q = self.duet_dist_q((self.embed(x_dist_q) * x_mask_q).permute(0, 2, 1))
            h_dist_d = self.duet_dist_d((self.embed(x_dist_d) * x_mask_d).permute(0, 2, 1))
            h_dist = self.duet_dist(h_dist_q.unsqueeze(-1) * h_dist_d)
        y_score = self.duet_comb(
            (h_local + h_dist) if ARCH_TYPE == 2 else (h_dist if ARCH_TYPE == 1 else h_local)) * self.scale
        return y_score

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def goInit(modelPath, data_file_dev, device):

    print_message('GoInit Start')

    model_dict = torch.load(modelPath)

    model_dict.eval()

    model_dict.to(device)

    reader_dev = DataReader(data_file_dev, 4, False)

    print_message('GoInit End')

    # feNames = ['sid', 'index', 'rel', 'label', 'query', 'doc']
    # df = pd.read_csv(data_file_dev, header=None, sep='\t', names=feNames)
    # df.sort_values(by=['sid', 'rel'], ascending=True, inplace=True)

    return model_dict, reader_dev


def goInfer(model, data_dev, device):

    res_dev = {}

    loop_cnt = 0
    while not is_complete:
        features = data_dev.get_minibatch()
        loop_cnt = loop_cnt + 1
        if ARCH_TYPE == 0:
            out = model(torch.from_numpy(features['local'][0]).to(device), None, None)
        elif ARCH_TYPE == 1:
            out = model(None, torch.from_numpy(features['dist_q']).to(device),
                      torch.from_numpy(features['dist_d'][0], torch.from_numpy(features['mask_q']).to(device),
                                       torch.from_numpy(features['mask_d'][0]).to(device)).to(device))
        else:
            out = model(torch.from_numpy(features['local'][0]).to(device),
                      torch.from_numpy(features['dist_q']).to(device),
                      torch.from_numpy(features['dist_d'][0]).to(device),
                      torch.from_numpy(features['mask_q']).to(device),
                      torch.from_numpy(features['mask_d'][0]).to(device))
        meta_cnt = len(features['meta'])

        if (loop_cnt % (1001) == 1):
            print_message("dev  meta_cnt:{} loop:{}".format(str(meta_cnt), str(loop_cnt)))

        out = out.data.cpu()
        for i in range(meta_cnt):
            q = features['meta'][i][0]
            d = features['meta'][i][1]
            if q not in res_dev:
                res_dev[q] = {}
            if d not in res_dev[q]:
                res_dev[q][d] = 0
            res_dev[q][d] += out[i][0]

        is_complete = (meta_cnt < MB_SIZE)

    print_message("eval count:{}".format(len(res_dev)))

    return res_dev


def getScore(sid, index, res_dev):
    score = DEFAULT_VAL
    if sid in res_dev.keys():
        dMap = res_dev[sid]

        # print_message("getSocre sid:{} index:{} dMap:{} ".format(sid, index, dMap.keys()))
        if index in dMap.keys():
            score = float(dMap[index])

    return score


def goEnvInit():
    print_message('Start goEnvInit')

    devName = "cpu"
    if torch.cuda.is_available():
        devName = "cuda:1"

    device = torch.device(devName)

    ts = datetime.datetime.utcnow().strftime("%b-%d-%H-%M-%S")
    print_message('Finished goEnvInit')

    return device, ts


# device = torch.device("cpu")  # torch.device("cpu"), if you want to run on CPU instead
ARCH_TYPE = 2
MAX_QUERY_TERMS = 20
MAX_DOC_TERMS = 200

NUM_HIDDEN_NODES = 128
TERM_WINDOW_SIZE = 3

POOLING_KERNEL_WIDTH_QUERY = MAX_QUERY_TERMS - TERM_WINDOW_SIZE + 1  # 20 - 3 + 1 = 18
POOLING_KERNEL_WIDTH_DOC = 100
NUM_POOLING_WINDOWS_DOC = (MAX_DOC_TERMS - TERM_WINDOW_SIZE + 1) - POOLING_KERNEL_WIDTH_DOC + 1  # (200 - 3 + 1) - 100 + 1 = 99

VOCAB_SIZE = 0
DROPOUT_RATE = 0.5

MB_SIZE = 1024

DEFAULT_VAL = -0.00001

DATA_DIR = 'data/'

DATA_FILE_VOCAB = os.path.join(DATA_DIR, "s_vocab.tsv")
DATA_EMBEDDINGS = os.path.join(DATA_DIR, "ft.vec.txt")
DATA_FILE_IDFS = os.path.join(DATA_DIR, "s_idf.norm.tsv")


def main(argv):
    if len(argv) != 4 :
        print("arg err: len = %d" %(len(argv)))
        print("arg like: python duet.py modelName dev_data savePath ")
        sys.exit(-1)
    else:
        print("go main : %d" %(len(argv)))
        print_message("modelPath:{} dev_data:{} savePath:{}".format(argv[1], argv[2], argv[3]))

    device, ts = goEnvInit()

    model, dev_data = goInit(argv[1], argv[2], device)

    dev_save = goInfer(model, dev_data, device)

    # reader_train, reader_dev, reader_eval, df_dev = goInit(DATA_FILE_TRAIN, DATA_FILE_DEV, DATA_FILE_EVAL)

    # res_dev = goRun(device, reader_train, reader_dev, reader_eval, ts, argv[1])

    # goEval(res_dev, df_dev)

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_deviceS"] = "0,1,2,3"
    main(sys.argv)