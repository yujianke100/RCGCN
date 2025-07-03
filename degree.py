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


def degree_collapsed_kcore(core, k, b):
    orig_core = deepcopy(core)
    for i in range(b):
        degree = nx.degree_centrality(core)
        node = max(dict(degree).items(), key=operator.itemgetter(1))[0]
        core.remove_node(node)
        G = nx.k_core(core, k)
        core = G
        #print("graph id: %d, all removed: %d" % (node, orig_core.number_of_nodes() - core.number_of_nodes() ))
        print(i,": ", orig_core.number_of_nodes() - core.number_of_nodes())
    #num = orig_core.number_of_nodes() - core.number_of_nodes()
    #print(b,"nodes removed: ",num)
    #print(num)

k = 190
fname = "/home/gxl/data/Flickr/Flickr_core_"+str(k)+".txt"
core = load_core_nx(fname)
degree_collapsed_kcore(core, k, b=2000)