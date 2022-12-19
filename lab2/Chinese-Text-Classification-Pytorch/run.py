# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import pandas as pd
import os


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, \
                        TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--hy', default='none', type=str, help='choose hypermeter in choosedic')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.hy == 'embsizev3':
        args.embedding = 'random'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    cache = train(config, model, train_iter, dev_iter, test_iter)
    # print(cache)
    pathstr = [r'train\loss', r'train\acc', r'dev\loss', r'dev\acc']
    cache_ind = ['train_loss', 'train_acc', 'dev_loss', 'dev_acc']

    if args.hy != 'none' and (args.model == 'TextCNN' or args.model == 'BiLSTM'):
        parent_path = os.path.join('results', args.model, args.hy)
        # print(cache)
        for i in range(len(cache_ind)):
            colums = [cache_ind[i]]
            data = cache[cache_ind[i]]
            print(colums, data[-1])

            save = pd.DataFrame(columns=colums, data=data)
            path = os.path.join(parent_path, pathstr[i])
            if not os.path.exists(path):
                os.makedirs(path)
            # changed
            choosedic = {
                'batchsize': str(config.batch_size),
                'lr': str(config.learning_rate),
                'dropout': str(config.dropout),
                'embsize': str(config.new_embed),
                'embsizev2': str(config.new_embed),
                'embsizev3': str(config.embed),
                'other': 'other'
            }
            if args.model == 'TextCNN':
                choosedic['numf'] = str(config.num_filters)
            else:
                choosedic['hiddensize'] = str(config.hidden_size)
            path = os.path.join(path, choosedic[args.hy])
            path = path.replace('.', '_') + '.csv'
            # if os.path.exists(path):
            #     os.remove(path)
            f1 = open(path, mode='w', newline='')
            save.to_csv(f1, encoding='gbk')
            f1.close()
    else:
        parent_path = os.path.join('results', 'all')
        # print(cache)
        for i in range(len(cache_ind)):
            colums = [cache_ind[i]]
            data = cache[cache_ind[i]]
            print(colums, data[-1])

            save = pd.DataFrame(columns=colums, data=data)
            path = os.path.join(parent_path, pathstr[i])
            if not os.path.exists(path):
                os.makedirs(path)
            # changed
            path = os.path.join(path, args.model) + '.csv'
            f2 = open(path, mode='w', newline='')
            save.to_csv(f2, encoding='gbk')
            f2.close()

