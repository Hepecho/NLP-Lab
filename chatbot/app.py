
# coding=utf-8
from flask import Flask, render_template, request, jsonify
import time
import threading
import jieba
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import unicodedata
import codecs
from io import open
import itertools
from model import EncoderRNN# 配置模型
from model import LuongAttnDecoderRNN
from dict import normalizeString
from model import GreedySearchDecoder# 配置模型
from model import evaluate
from dict import loadPrepareData
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile)


model_name = 'cb_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# 从哪个checkpoint恢复，如果是None，那么从头开始训练。
loadFilename = './res/cb_model/cornell movie-dialogs corpus/2-2_500/4000_checkpoint.tar'  # change to 4000
"""
定义心跳检测函数
"""
def heartbeat():
    print (time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()
timer = threading.Timer(60, heartbeat)
timer.start()

"""
ElementTree在 Python 标准库中有两种实现。
一种是纯 Python 实现例如 xml.etree.ElementTree ，
另外一种是速度快一点的 xml.etree.cElementTree 。
 尽量使用 C 语言实现的那种，因为它速度更快，而且消耗的内存更少
"""


app = Flask(__name__,static_url_path="/static") 

@app.route('/message', methods=['POST'])

#"""定义应答函数，用于获取输入信息并返回相应的答案"""
def reply():
#从请求中获取参数信息
    req_msg = request.form['msg']
    print(req_msg)
#将语句使用结巴分词进行分词
    #req_msg=" ".join(req_msg)

    req_msg = normalizeString(req_msg)
    if loadFilename:
    # 如果训练和加载是一条机器，那么直接加载
        checkpoint = torch.load(loadFilename)
    # 否则比如checkpoint是在GPU上得到的，但是我们现在又用CPU来训练或者测试，那么注释掉下面的代码
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
# 初始化word embedding
# 初始化word embedding
    embedding = nn.Embedding(voc.num_words, hidden_size)
    #调用decode_line对生成回答信息
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
# 初始化encoder和decoder模型
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
# 使用合适的设备
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    encoder.eval()
    decoder.eval()
# 构造searcher对象
    searcher = GreedySearchDecoder(encoder, decoder)

# 测试

    res_msg = evaluate(encoder, decoder, searcher, voc, req_msg)
    print(res_msg)
            # 去掉EOS后面的内容
    words = []
    for word in res_msg:
        if word == 'EOS':
            break
        elif word != 'PAD':
            words.append(word)
            words.append(' ')


    #res_msg = execute.predict(req_msg)
    #将unk值的词用微笑符号袋贴
    #res_msg = res_msg.replace('_UNK', '^_^')
    #res_msg=res_msg.strip()
    
    # 如果接受到的内容为空，则给出相应的回复
    if res_msg == ' ':
      res_msg = '请与我聊聊天吧'

    return jsonify( { 'text': ''.join(words)} )

"""
jsonify:是用于处理序列化json数据的函数，就是将数据组装成json格式返回

http://flask.pocoo.org/docs/0.12/api/#module-flask.json
"""
@app.route("/")
def index(): 
    return render_template("index.html")
'''
'''
# 启动APP
if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8808) 
