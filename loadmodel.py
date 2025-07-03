import sys
import math
import kcore
import kcore_SCGCN
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

def main(args):
    n_hide1 = args.n_hid1
    n_hide2 = args.n_hid2
    n_output = args.n_output
    input_data = args.input_data
    model_dir = args.model_dir
    steps = args.steps
    lr = args.learning_rate
    weight_decay = args.weight_decay
    imagination = args.imagination
    p = args.p
    verbose = args.verbose
    b = args.b
    k = args.k

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    x = np.loadtxt('/home/gxl/data/facebook/feature8_facebook_core_20.txt', delimiter=',')
    x = torch.FloatTensor(x)
    n_feature=x.size(1)
    core = load_core_pyg(input_data,x)
    G = kcore.Graph()
    G.loadUndirGraph(input_data)
    g = kcore_SCGCN.Graph2()
    g.loadUndirGraph(input_data)
    g.KCoreCollapseDominate(b)
    un_dominated = np.array(g.getUnDominated())#un_dominated为剪枝后剩下的节点的id
    un_dominated = torch.from_numpy(un_dominated)#类型转换为tensor
    #un_dominated = un_dominated.cuda()
    state_dict = torch.load(args.model_dir)
    model = REINFORCE(n_feature, n_hide1, n_hide2, n_output, k, b, imagination)
    model.load_state_dict(state_dict)
    prob, indices = model(core, un_dominated, p)
    idx = indices.cpu()
    idx = idx.numpy()
    reward = G.GetFollowerNum(args.k, idx)  # indices为list[]，或者np.array
    print(reward)



if __name__ == "__main__":
    parser = ArgumentParser("gcnRL", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    # Model settings
    parser.add_argument("--n_hid1", default=64, type=int,
                        help="first layer of GCN: number of hidden units")  # options [64, 128, 256]
    parser.add_argument("--n_hid2", default=64, type=int,
                        help="second layer of GCN: number of hidden units")  # options [64, 128, 256]
    parser.add_argument("--n_output", default=1, type=int,
                        help="output layer of GCN: number of output units")  # options [64, 128, 256]
    parser.add_argument("--model_dir", type=str, default="model_save/gcnRL_model_facebook_core_20_b11.pt")

    # Training settings
    parser.add_argument("--steps", default=5000, type=int)  # options:  (1000, 2000, ... 40000)
    parser.add_argument("--learning_rate", default=0.001, type=float)  # options [1e-3, 1e-4]
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--imagination", default=False, type=bool, help="whether use random sample")
    parser.add_argument("--p", default=0.5, type=float, help="the probability of using imagination in each iteration")
    # Others
    parser.add_argument("--input_data", default="/home/gxl/data/facebook/facebook_core_20.txt", help="Input graph data ")
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--k", default=20, type=int, help="the k core to be collesped")  # options [20, 30, 40]
    parser.add_argument("--b", default=11, type=int, help="the result set size")


    args = parser.parse_args()
    if args.verbose:
        print(args)
    main(args)



