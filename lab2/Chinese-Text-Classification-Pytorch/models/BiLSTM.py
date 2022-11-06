# _*_ coding :utf-8 _*_
# time: 2022/10/26/026 19:05:48
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'BiLSTM'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活  H
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                          # mini-batch大小 H
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 8e-4                                       # 学习率  H
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度  H
        self.new_embed = 64
        self.hidden_size = 128                                          # lstm隐藏层  H
        self.num_layers = 2                                             # lstm层数


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # self.fc_resize = nn.Linear(config.embed, config.new_embed)
        # if config.embed > config.new_embed:
            # self.conv_resize = nn.Conv2d(1, 1, (1, config.embed - config.new_embed + 1))
            # 保持channels和H不变，只改变W，由于stride = 1 padding = 0, 所以W'=W - (kernel[1] - 1)推导得kernel[1]
        # else:
            # self.conv_resize = nn.Conv2d(1, 1, (1, 1), padding=(0, int((config.new_embed-config.embed)/2)))
        # 如果增加resize则将下面config.embed改为config.new_emb
        self.bilstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(2 * config.hidden_size, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        # out = out.unsqueeze(1)
        # [batch, 1, config.n_vocab, config.embed] [64, 1, 32, 300]
        # out = self.fc_resize(out)
        # out = self.conv_resize(out)
        # out = out.squeeze(1)
        out, (hidden, _) = self.bilstm(out)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        out = self.dropout(hidden)
        out = self.fc(out)
        return out