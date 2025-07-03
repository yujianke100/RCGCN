#### Imports ####
import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from DataProcess import *

# with open('karate.adjlist', 'r+', encoding='utf-8') as f:
#     s = [i[:-1].split(' ') for i in f.readlines()]
# i=0
# for item in s:
#     s[i].pop(0)
#     s[i] = list(map(int, s[i]))
#     for j in range(len(s[i])):
#         s[i][j]=s[i][j]-1
#     i+=1
# adj_list=s

adj_list=adjlist_to_list('temp_core_60.adjlist')
for i in adj_list:
    print(i)
# adj_list = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2], [5, 6], [4, 6], [4, 5], [1, 3]]
size_vertex = len(adj_list)  # number of vertices

#### Hyperparameters ####

w = 3  # window size
d = 8  # embedding size
y = 200  # walks per vertex
t = 6  # walk length
lr = 0.025  # learning rate

# v = [0, 1, 2, 3, 4, 5, 6, 7]  # labels of available vertices
v=[]
for i in range(size_vertex):
    v.append(i)

#### Random Walk ####

def RandomWalk(node, t):
    walk = [node]  # Walk starts from this node

    for i in range(t - 1):
        node = adj_list[node][random.randint(0, len(adj_list[node]) - 1)]
        walk.append(node)

    return walk


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.phi = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))
        self.phi2 = nn.Parameter(torch.rand((d, size_vertex), requires_grad=True))

    def forward(self, one_hot):
        hidden = torch.matmul(one_hot, self.phi)
        out = torch.matmul(hidden, self.phi2)
        return out


model = Model()
model.cuda()

def skip_gram(wvi, w):
    for j in range(len(wvi)):
        for k in range(max(0, j - w), min(j + w, len(wvi))):

            # generate one hot vector
            one_hot = torch.zeros(size_vertex)
            one_hot[wvi[j]] = 1

            one_hot = one_hot.cuda()
            model.cuda()
            out = model(one_hot)
            out = out.cuda()
            loss = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
            loss = loss.cuda()
            loss.backward()

            for param in model.parameters():
                param.data.sub_(lr * param.grad)
                param.grad.data.zero_()


for i in range(y):
    random.shuffle(v)
    for vi in v:
        wvi = RandomWalk(vi, t)
        skip_gram(wvi, w)

print(model.phi)


#### Hierarchical Softmax ####

def func_L(w):
    """
    Parameters
    ----------
    w: Leaf node.

    Returns
    -------
    count: The length of path from the root node to the given vertex.
    """
    count = 1
    while (w != 1):
        count += 1
        w //= 2

    return count


# func_n returns the nth node in the path from the root node to the given vertex
def func_n(w, j):
    li = [w]
    while (w != 1):
        w = w // 2
        li.append(w)

    li.reverse()

    return li[j]


def sigmoid(x):
    out = 1 / (1 + torch.exp(-x))
    return out


class HierarchicalModel(torch.nn.Module):

    def __init__(self):
        super(HierarchicalModel, self).__init__()
        self.phi = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))
        self.prob_tensor = nn.Parameter(torch.rand((2 * size_vertex, d), requires_grad=True))

    def forward(self, wi, wo):
        one_hot = torch.zeros(size_vertex)
        one_hot[wi] = 1
        one_hot = one_hot.cuda()
        w = size_vertex + wo
        h = torch.matmul(one_hot, self.phi)
        h=h.cuda()
        p = torch.tensor([1.0])
        p=p.cuda()
        for j in range(1, func_L(w) - 1):
            mult = -1
            if (func_n(w, j + 1) == 2 * func_n(w, j)):  # Left child
                mult = 1

            p = p * sigmoid(mult * torch.matmul(self.prob_tensor[func_n(w, j)], h))
            p=p.cuda()

        return p


hierarchicalModel = HierarchicalModel()
hierarchicalModel.cuda()

def HierarchicalSkipGram(wvi, w):
    for j in range(len(wvi)):
        for k in range(max(0, j - w), min(j + w, len(wvi))):
            # generate one hot vector

            prob = hierarchicalModel(wvi[j], wvi[k])
            prob = prob.cuda()
            loss = - torch.log(prob)
            loss = loss.cuda()
            loss.backward()
            for param in hierarchicalModel.parameters():
                param.data.sub_(lr * param.grad)
                param.grad.data.zero_()


for i in range(y):
    random.shuffle(v)
    for vi in v:
        wvi = RandomWalk(vi, t)
        HierarchicalSkipGram(wvi, w)

for i in range(8):
    for j in range(8):
        print((hierarchicalModel(i, j).item() * 100) // 1, end=' ')
    print(end='\n')

print("------------------------------------------------")
print(hierarchicalModel.phi)
print("------------------------------------------------")
print("model的divice：")
print(next(model.parameters()).device)
hierarchicalModel.cpu()
feature=hierarchicalModel.phi.detach().numpy()
print(feature)
np.savetxt(fname='feature_temp_core_60.txt', X=feature, delimiter=',')
