# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
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

        self.dropout = 0.2                                             # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 64                                            # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.new_embed = 64                                            # 用卷积核降维后的字向量维度/卷积核数量(channels数)
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # [batch, 1, config.n_vocab, config.embed]
        # 1.conv
        # if config.embed > config.new_embed:
            # self.conv_resize = nn.Conv2d(1, 1, (1, config.embed - config.new_embed + 1))
            # 保持channels和H不变，只改变W，由于stride = 1 padding = 0, 所以W'=W - (kernel[1] - 1)推导得kernel[1]
        # else:
            # self.conv_resize = nn.Conv2d(1, 1, (1, 1), padding=(0, int((config.new_embed-config.embed)/2)))
        # 2.fc
        # self.fc_resize = nn.Linear(config.embed, config.new_embed)
        # 如果增加resize则将下面config.embed改为config.new_emb
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        # x = [batch, num_filter, sent len - filter_sizes+1]
        # 有几个filter_sizes就有几个x
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        # [batch, config.n_vocab, config.embed] [64, 32, 300]
        out = out.unsqueeze(1)
        # [batch, 1, config.n_vocab, config.embed] [64, 1, 32, 300]
        # out = self.conv_resize(out)  # 1.conv
        # out = self.fc_resize(out)  # 2.fc
        # [batch, 1, config.n_vocab, config.new_embed] [64, 1, 32, 2]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # [batch, num_filter * len(filter_sizes) [64, 768]
        out = self.dropout(out)
        # [batch, num_filter * len(filter_sizes)] [64, 768]
        # 把 len(filter_sizes)个卷积模型concate起来传到全连接层。
        out = self.fc(out)
        # [64, 4]
        return out
