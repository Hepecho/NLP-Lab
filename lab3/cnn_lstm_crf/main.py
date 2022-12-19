import argparse
import numpy as np
import torch
import os
from data import build_corpus
from utils import extend_maps, prepocess_data_for_crf
from evaluate import train_and_eval
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bilstm', help='choose base model, cnn or bilstm')
    parser.add_argument("--crf", default=False, action='store_true', help='add crf')
    parser.add_argument('--use_w2v', default=False, action='store_true', help='use pretrained vectors')
    parser.add_argument('--hy', type=str, default=None, help='hyperparamter')
    opt = parser.parse_args()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id, word_emb = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    print("正在训练评估{}{}{}模型...".format(opt.model, '+CRF' if opt.crf else '', '+w2v' if opt.use_w2v else ''))
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    tag2id = extend_maps(tag2id, for_crf=opt.crf)
    # crf还需要额外的一些数据处理
    if opt.crf:
        train_word_lists, train_tag_lists = prepocess_data_for_crf(
            train_word_lists, train_tag_lists
        )
        dev_word_lists, dev_tag_lists = prepocess_data_for_crf(
            dev_word_lists, dev_tag_lists
        )
        test_word_lists, test_tag_lists = prepocess_data_for_crf(
            test_word_lists, test_tag_lists, test=True
        )

    _, cache = train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id, tag2id, word_emb, opt)

    # cache['train_loss'].append(1.14514)
    name = opt.model
    if opt.crf:
        name = name + '+crf'
    if opt.use_w2v:
        name = name + '+w2v'
    if opt.hy:
        name = name + '+' + opt.hy
    path = os.path.join('results', name)
    # print(cache)

    colums = ['dev_loss']
    data = cache['dev_loss']
    print(colums, data[-1])

    save = pd.DataFrame(columns=colums, data=data)

    # changed
    # if os.path.exists(path):
    #     os.remove(path)
    f1 = open(path + '.csv', mode='w', newline='')
    save.to_csv(f1, encoding='gbk')
    f1.close()

