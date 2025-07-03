import random
import argparse
import networkx as nx
import numpy as np
from DataProcess import *
from gensim.models import Word2Vec
from tqdm import tqdm
#from base_model import BaseModel
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from torch_geometric.utils import to_networkx
from typing import Optional, Type, Any
import torch.nn as nn


class BaseModel(nn.Module):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        """Build a new model instance."""
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = self.__class__.__name__
        self.loss_fn = None
        self.evaluator = None

    def _forward_unimplemented(self, *input: Any) -> None:  # abc warning
        pass

    def forward(self, *args):
        raise NotImplementedError

    def predict(self, data):
        return self.forward(data)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn



class DeepWalk(BaseModel):
    r"""The DeepWalk model from the `"DeepWalk: Online Learning of Social Representations"
    <https://arxiv.org/abs/1403.6652>`_ paper

    Args:
        hidden_size (int) : The dimension of node representation.
        walk_length (int) : The walk length.
        walk_num (int) : The number of walks to sample for each node.
        window_size (int) : The actual context size which is considered in language model.
        worker (int) : The number of workers for word2vec.
        iteration (int) : The number of training iteration in word2vec.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--walk-length", type=int, default=80,
                            help="Length of walk per source. Default is 80.")
        parser.add_argument("--walk-num", type=int, default=40,
                            help="Number of walks per source. Default is 40.")
        parser.add_argument("--window-size", type=int, default=5,
                            help="Window size of skip-gram model. Default is 5.")
        parser.add_argument("--worker", type=int, default=10,
                            help="Number of parallel workers. Default is 10.")
        parser.add_argument("--iteration", type=int, default=10,
                            help="Number of iterations. Default is 10.")
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args) -> "DeepWalk":
        return cls(
            args.hidden_size,
            args.walk_length,
            args.walk_num,
            args.window_size,
            args.worker,
            args.iteration,
        )

    def __init__(self, dimension, walk_length, walk_num, window_size, worker, iteration):
        super(DeepWalk, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration

    def forward(self, graph, embedding_model_creator=Word2Vec, return_dict=False):#graph为pyg类型的图
        nx_g = to_networkx(graph)
        self.G = nx_g
        walks = self._simulate_walks(self.walk_length, self.walk_num)
        walks = [[str(node) for node in walk] for walk in walks]
        print("training word2vec...")
        model = embedding_model_creator(
            walks,
            vector_size=self.dimension,
            #size=self.dimension,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.worker,
            epochs=self.iteration,
            #iter=self.iteration,
        )
        id2node = dict([(vid, node) for vid, node in enumerate(nx_g.nodes())])
        embeddings = np.asarray([model.wv[str(id2node[i])] for i in range(len(id2node))])

        if return_dict:
            features_matrix = dict()
            for vid, node in enumerate(nx_g.nodes()):
                features_matrix[node] = embeddings[vid]
        else:
            features_matrix = np.zeros((graph.num_nodes, embeddings.shape[1]))
            nx_nodes = nx_g.nodes()
            features_matrix[nx_nodes] = embeddings[np.arange(graph.num_nodes)]
        return features_matrix

    def _walk(self, start_node, walk_length):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            k = int(np.floor(np.random.rand() * len(cur_nbrs)))
            walk.append(cur_nbrs[k])
        return walk

    def _simulate_walks(self, walk_length, num_walks):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("node number:", len(nodes))
        print("generating random walks...")
        for walk_iter in tqdm(range(num_walks)):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._walk(node, walk_length))
        return walks


def main(args):
    dimension = args.hidden_size
    walk_length = args.walk_length
    walk_num = args.walk_num
    window_size = args.window_size
    worker = args.worker
    iteration = args.iteration

    fname = 'data/Flickr/Flickr_core_200.txt'
    node_num = 2130
    x = torch.eye(node_num)
    graph = load_core_pyg(fname, x)
    dp = DeepWalk(dimension, walk_length, walk_num, window_size, worker, iteration)
    features_matrix = dp(graph)
    np.savetxt(fname='data/Flickr/feature8_Flickr_core_200.txt', X=features_matrix, delimiter=',')


if __name__ == "__main__":
    parser = ArgumentParser("dpwk", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--walk_length", type=int, default=80,
                        help="Length of walk per source. Default is 80.")
    parser.add_argument("--walk_num", type=int, default=40,
                        help="Number of walks per source. Default is 40.")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Window size of skip-gram model. Default is 5.")
    parser.add_argument("--worker", type=int, default=10,
                        help="Number of parallel workers. Default is 10.")
    parser.add_argument("--iteration", type=int, default=10,
                        help="Number of iterations. Default is 10.")
    parser.add_argument("--hidden_size", type=int, default=8)

    args = parser.parse_args()

    main(args)
