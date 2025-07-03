import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import sys
import math
import kcore
# import kcore_SCGCN
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
from GCN import GCNNet
from model import REINFORCE
from DataProcess import *
import pdb
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
# setup_seed(20)


def myloss(reward, prob, baseline_reward):
    prob_sum=prob.sum()#;print("prob_sum:",prob_sum)
    loss = (reward - baseline_reward)*torch.log(prob_sum)
    return loss


def train(args, model, core, optimizer, baseline_reward, G, un_dominated, p):
    if args.cuda:
        model.cuda()

    maxid=0
    maxreward=0
    for step in range(args.steps):
        model.train()
        if args.cuda:
            core = core.cuda()

        optimizer.zero_grad()
        prob, indices = model(core, un_dominated, p)
        idx = indices.cpu()
        idx = idx.numpy()
        reward = G.GetFollowerNum(args.k, idx)#indices为list[]，或者np.array
        follower_exact = np.array(G.Get_follower_exact())
        un_dominated = un_dominated.cpu()
        un_dominated = un_dominated.numpy()
        un_dominated = np.setdiff1d(un_dominated, follower_exact)
        un_dominated = torch.from_numpy(un_dominated)
        un_dominated = un_dominated.cuda()

        loss = myloss(reward, prob, baseline_reward)
        loss = -loss
        print("                                   ",step,":",reward,"len(un_dominated):",un_dominated.numel(),"idx:",idx)
        if reward >maxreward:
            maxid = step
            maxreward = reward

        loss.backward()
        optimizer.step()
    print("                                   The best:")
    print("                                   ",maxid,":",maxreward)
    return model


def main(args):
    n_hide1 = args.n_hid1
    n_hide2 = args.n_hid2
    n_output = args.n_output
    input_data = args.input_data
    # model_dir = args.model_dir
    steps = args.steps
    lr = args.learning_rate
    weight_decay = args.weight_decay
    imagination = args.imagination
    p = args.p
    verbose = args.verbose
    b = args.b
    k = args.k

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    x = np.loadtxt('/home/shinshi/xulu_information_sciences/data/WormNet/feature8_WormNet_core_130.txt', delimiter=',')
    x = torch.FloatTensor(x)
    n_feature=x.size(1)
    core = load_core_pyg(input_data,x)
    G = kcore.Graph()
    G.loadUndirGraph(input_data)
    G_tmp = kcore.Graph()
    G_tmp.loadUndirGraph(input_data)
    un_dominated = np.array(G.Get_un_dominated(k))#un_dominated为剪枝后剩下的节点的id
    un_dominated = torch.from_numpy(un_dominated)#类型转换为tensor
    un_dominated = un_dominated.cuda()
    print("un_dominated:",un_dominated.numel())
    model = REINFORCE(n_feature, n_hide1, n_hide2, n_output, k, b-1, imagination)
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=args.weight_decay)
    start_time = datetime.datetime.now()  # 开始时间点
    greedy_rslt = G_tmp.Greedy(k,b)
    end_time = datetime.datetime.now()  # 开始时间点
    print("Greedy time consuming：",end_time-start_time)
    print("greedy:",greedy_rslt)
    if b==1:
        baseline_reward = greedy_rslt
    else:
        baseline_reward = G.Greedy(k, b-1)
    #baseline_reward = 239
    print("                                   baseline_reward:",baseline_reward)
    model = train(args, model, core, optimizer ,baseline_reward, G, un_dominated, p)
    #torch.save(model.state_dict(), args.model_dir)
    print("                                   baseline_reward:",baseline_reward)

    model.eval()
    print("len(un_dominated):",un_dominated.numel())
    start_time = datetime.datetime.now()  # 开始时间点
    prob, indices = model(core, un_dominated, p)
    idx = indices.cpu()
    idx = idx.numpy()
    reward = G.GetFollowerNum(args.k, idx)
    follower_exact = np.array(G.Get_follower_exact())
    un_dominated = un_dominated.cpu()
    un_dominated = un_dominated.numpy()
    un_dominated = np.setdiff1d(un_dominated, follower_exact)
    un_dominated = torch.from_numpy(un_dominated)
    un_dominated = un_dominated.cuda()
    #print("len(un_dominated):", un_dominated.numel())
    prob, indices = model(core, un_dominated, p, test=True)
    end_time = datetime.datetime.now()  # 开始时间点
    print("Our time consuming：",end_time-start_time)

    idx = indices.cpu()
    idx = idx.numpy()
    reward = G.GetFollowerNum(args.k, idx)
    print("b reward:",reward)
    print("idx:",idx)




if __name__ == "__main__":
    parser = ArgumentParser("gcnRL", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    # Model settings
    parser.add_argument("--n_hid1", default=64, type=int,
                        help="first layer of GCN: number of hidden units")  # options [64, 128, 256]
    parser.add_argument("--n_hid2", default=64, type=int,
                        help="second layer of GCN: number of hidden units")  # options [64, 128, 256]
    parser.add_argument("--n_output", default=1, type=int,
                        help="output layer of GCN: number of output units")  # options [64, 128, 256]
    # parser.add_argument("--model_dir", type=str, default="model_save/gcnRL_model_Flickr_core_190_b++.pt")

    # Training settings
    parser.add_argument("--steps", default=1000, type=int)  # options:  (1000, 2000, ... 40000)
    parser.add_argument("--learning_rate", default=0.001, type=float)  # options [1e-3, 1e-4]
    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
    parser.add_argument('--weight_decay', type=float, default=0,help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--imagination", default=False, type=bool, help="whether use random sample")
    parser.add_argument("--p", default=0.5, type=float, help="the probability of using imagination in each iteration")
    # Others
    parser.add_argument("--input_data", default="/home/shinshi/xulu_information_sciences/data/WormNet/WormNet_core_130.txt", help="Input graph data ")
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--k", default=130, type=int, help="the k core to be collesped")  # options [20, 30, 40]
    parser.add_argument("--b", default=10, type=int, help="the result set size")


    args = parser.parse_args()
    if args.verbose:
        print(args)
    main(args)
