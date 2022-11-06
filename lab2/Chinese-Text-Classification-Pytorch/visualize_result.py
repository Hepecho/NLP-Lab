import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import time
import numpy as np
import os
import argparse
from scipy.interpolate import make_interp_spline


plt.rcParams['axes.unicode_minus'] = False
colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
color_i = 0  # color
# mcolors.TABLEAU_COLORS可以得到一个字典，可以选择TABLEAU_COLORS,CSS4_COLORS等颜色组

parser = argparse.ArgumentParser(description='visualize_result')
parser.add_argument('--model', default='TextCNN', type=str, required=True, help='choose a model')
parser.add_argument('--ind', default='loss', type=str, help='choose a index')
parser.add_argument('--set', default='train', type=str, help='choose a setting')
parser.add_argument('--hy', default='batchsize', type=str, help='Hypermeter')
args = parser.parse_args()


def read_csv_xy(path):
    '''
    此函数用来读取csv文档并且根据epochs返回对应指标
    '''
    exampleFile = open(path+'.csv')  # 打开csv文件
    exampleReader = csv.reader(exampleFile)  # 读取csv文件
    exampleData = list(exampleReader)  # csv数据转换为列表
    # print(exampleData)
    length_zu = len(exampleData)  # 得到数据行数
    length_yuan = len(exampleData[0])  # 得到每行长度

    # 建立两个空list
    x = []  # list()
    y = []  # list()

    for i in range(1, length_zu):  # 从第二行开始读取
        x.append(int(exampleData[i][0]) + 1)  # 将第一列数据从第二行读取到最后一行赋给列表x
        y.append(float(exampleData[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表y
    exampleFile.close()
    return x, y


def add_line(path, name, args):
    global color_i
    x, y = read_csv_xy(path)

    plt.figure(1)
    lable = name
    color = mcolors.TABLEAU_COLORS[colors[color_i]]
    color_i += 1

    # 点线图
    plt.plot(x, y, linestyle='-', color=color, label=lable)


if __name__ == '__main__':
    if args.model == 'all':
        head = '{}({} set)'.format('all models', args.set)
        label_y = args.ind
        parent_path = os.path.join('results', args.model, args.set, args.ind)
        # 图片保存
        fig_path = os.path.join('results', args.model, args.set, label_y + '.jpg')
    else:
        head = '{}({} set)'.format(args.hy, args.set)
        label_y = '{} {}'.format(args.model, args.ind)
        parent_path = os.path.join('results', args.model, args.hy, args.set, args.ind)
        # 图片保存
        fig_path = os.path.join('results', args.model, args.hy, args.set, label_y + '.jpg')
    print('parent_path: ' + parent_path)
    names = os.listdir(parent_path)
    for name in names:
        name = name.replace('.csv', '')

        path = os.path.join(parent_path, name)
        if args.model != 'all':
            name = name.replace('_', '.')
        add_line(path, name, args)

    # 点图
    plt.title(head)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(label_y)
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    plt.gca().xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.xlim(0.5, 20.5)

    print('fig_path: ' + fig_path)
    plt.savefig(fig_path)
    print("picture is saved")
    plt.show()
