import numpy as np
import networkx as nx
import os
import torch
import random
#import kcore
import scipy.sparse as sp
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_networkx
from DataProcess import *

fname = 'temp_core_50.txt'
graph_to_adjlist(fname)