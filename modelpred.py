# 复数神经网络 需要加载ComplexPyTorch库


'''
数据集方案：
方案一：采用输入及槽均随机方式采集的信息（10000*2数据）进行拟合：是否降维处理过拟合问题存在
方案二：采用针对120训练样本生成输入，随机槽的方式（120*100=12000数据）进行拟合：过拟合问题缓解
方案三：针对特定输入，即仅改变槽的大小（100数据）进行拟合：过拟合问题存在
方案四：待讨论......

'''

import torch
import numpy as np
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_normalize, complex_dropout
from model import ComplexNet
from sklearn import preprocessing

def prediction(data1, data2):

    data1 = np.array(data1)
    data2 = np.array(data2)
    data2 = np.transpose(data2)

    min_max_scaler = preprocessing.MinMaxScaler()
    data2 = min_max_scaler.fit_transform(data2)
    data2 = np.transpose(data2)
    data1 = complex_normalize(data1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ComplexNet(data_size = 673,
                   data_slot = 50,
                   neuron1 = 1000,
                   neuron2 = 100,
                   neuron3 = 1200).to(device)
    model.load_state_dict(torch.load('net_3.pt'))

    data1 = torch.from_numpy(data1).to(device).type(torch.complex64)
    data2 = torch.from_numpy(data2).to(device).type(torch.complex64)

    model.eval()
    pred = model(data1, data2)
    pred = pred.cpu().detach().numpy()
    pred = np.ascontiguousarray(pred)


    return pred