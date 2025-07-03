import math
from DataProcess import *
import pdb
import networkx as nx
import numpy as np
import scipy as sp
import os
import random
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from scipy.stats import entropy
from copy import deepcopy
import operator
from random import choice



def centrality_collapsed_kcore(core, k, b):
    orig_core = deepcopy(core)
    for i in range(b):
        centrality = nx.eigenvector_centrality_numpy(core)
        node = max(dict(centrality).items(), key=operator.itemgetter(1))[0]
        core.remove_node(node)
        G = nx.k_core(core, k)
        core = G
        #print("graph id: %d, all removed: %d" % (node, orig_core.number_of_nodes() - core.number_of_nodes() ))
        print(i,": ", orig_core.number_of_nodes() - core.number_of_nodes())



k = 130
fname = "/home/gxl/data/Wiki-Talk/Wiki-Talk_core_"+str(k)+".txt"
#fname = "C:/Users/Administrator/Desktop/data/facebook/facebook_core_"+str(k)+".txt"
core = load_core_nx(fname)
centrality_collapsed_kcore(core, k, b=2000)



























