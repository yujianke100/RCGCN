import sys
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
from GCN import GCNNet
import pdb
# import kcore_SCGCN


class Policy(nn.Module):
    def __init__(self, n_feature, n_hide1,n_hide2, n_output):
        super(Policy, self).__init__()
        self.GCN = GCNNet(n_feature, n_hide1,n_hide2, n_output)

    def forward(self, inputs):
        return self.GCN(inputs)





class REINFORCE(nn.Module):
    def __init__(self, n_feature, n_hide1, n_hide2, n_output, k, b, imagination):
        super(REINFORCE, self).__init__()
        self.n_feature=n_feature
        self.n_hide1=n_hide1
        self.n_hide2 = n_hide2
        self.n_output=n_output
        self.k=k
        self.b=b
        self.imagination = imagination
        self.policy = Policy(n_feature, n_hide1,n_hide2, n_output)


    def forward(self, inputs, un_dominated, p, test=False):#解空间有剪枝时使用这段代码
        output_embedding = self.policy(inputs)
        embedding_target = torch.index_select(output_embedding, 0, un_dominated)#将剪枝后的结果挑出来，un_dominated为剪枝后剩下的节点的id
        probs = F.softmax(embedding_target, dim=0)
        probs = torch.squeeze(probs)#;print("probs:\n",probs.size())
        if self.imagination:
            arr = [0, 1]
            t = np.random.choice(arr, 1, p=[1-p, p])
            if t==1: #if t equals to 1,implement random sample
                idx = torch.multinomial(probs, self.b)
                prob = torch.index_select(probs, 0, idx)
                indices = torch.index_select(un_dominated, 0, idx)
                return prob, indices

            else:    #if t does'nt equals to 1,implement topk
                prob, indices = probs.topk(self.b, dim=0, largest=True, sorted=True)  # 取出probs前b大元素及其原索引(需要probs为一维tensor)
                indices = torch.squeeze(indices)
                indices = torch.index_select(un_dominated, 0, indices)
                return prob, indices

        else:
            if test:
                self.b += 1
            if self.b == 0:
                self.b = 1
            prob, indices = probs.topk(self.b, dim=0, largest=True, sorted=True)#取出probs前b大元素及其原索引(需要probs为一维tensor)
            indices = torch.squeeze(indices)
            indices = torch.index_select(un_dominated, 0, indices)
            return prob, indices



























