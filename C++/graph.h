
#pragma once
#ifndef _GRAPH_H
#define _GRAPH_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <string.h>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>

using namespace std;

typedef int NodeID;
typedef vector<NodeID> Adjlist;
#define inf 99999999;

class Graph {
public:
    map <NodeID, NodeID> node_dict;
    vector <Adjlist> G_IN;
    vector <Adjlist> G_OUT;
    vector <int> degree;
    vector <int> greedy_rslt_ID;        //存储greedy算法求取出的节点的ID
    vector <int> follower_exact;        //存储GetFollowerNum方法求出的除idx本身外的follower
    int greedy_rslt_flnum = 0;          //存储greedy算法求取出的follower数量
    vector <int> un_dominated;          //存储初始剪枝后解空间

    long nodenum;
    long edgenum;

    Graph();
    ~Graph();

    void loadUndirGraph(const string filename);
    vector <int> RemoveNode(int k, NodeID id);
    int GetFollowerNum(int k, vector<NodeID>idx);
    vector <int> Get_follower_exact();
    vector <int> FindBest(int k, vector <int> deg);
    int Greedy(int k, int b);
    void Compute_un_dominated(int k);
    vector <int> Get_un_dominated(int k);
};
#endif