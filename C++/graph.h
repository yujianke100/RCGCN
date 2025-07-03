
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
    vector <int> greedy_rslt_ID;        //�洢greedy�㷨��ȡ���Ľڵ��ID
    vector <int> follower_exact;        //�洢GetFollowerNum��������ĳ�idx�������follower
    int greedy_rslt_flnum = 0;          //�洢greedy�㷨��ȡ����follower����
    vector <int> un_dominated;          //�洢��ʼ��֦���ռ�

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