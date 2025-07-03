import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import os

class GCNNet(torch.nn.Module):
    def __init__(self, n_feature, n_hide1,n_hide2, n_output):
        '''
        :param n_feature: 每个节点属性的维度
        :param n_hide: 隐含层维度
        :param n_output: 输出层维度
        '''
        super(GCNNet, self).__init__()
        self.n_feature = n_feature
        self.n_output = n_output
        self.conv1 = GCNConv(n_feature, n_hide1)
        self.conv2 = GCNConv(n_hide1, n_hide2)
        self.conv3 = GCNConv(n_hide2, n_output)

    # def forward(self, data):#解空间无剪枝时使用这段代码
    #     """
    #     :param data:
    #     :return:
    #     """
    #     # data是输入到网络中的训练数据
    #     # 获取节点的属性信息和边的连接信息
    #     x, edge_index = data.x, data.edge_index
    #     x = self.conv1(x, edge_index)
    #     x = F.relu(x)
    #     x = self.conv2(x, edge_index)
    #     output = F.softmax(x, dim=0)
    #     return output

    def forward(self, data):#解空间有剪枝时使用这段代码
        """
        :param data:
        :return:
        """
        # data是输入到网络中的训练数据
        # 获取节点的属性信息和边的连接信息
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.tanh(x)
        x = self.conv3(x, edge_index)
        return x

