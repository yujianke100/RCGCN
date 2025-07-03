
#include "graph.h"
#include <iostream>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define thread_count 30


Graph::Graph() {
    nodenum = edgenum = 0;
}

Graph::~Graph() {
}

void Graph::loadUndirGraph(const string filename) {
    ifstream infile;
    infile.open(filename);

    string line;
    string fid;
    string tid;

    string node_num;
    string edge_num;

    getline(infile, line);
    stringstream ss(line);
    ss >> node_num;
    nodenum = atol(node_num.c_str());
    ss >> edge_num;
    edgenum = atol(edge_num.c_str());

    G_IN.resize(nodenum);
    G_OUT.resize(nodenum);
    degree.resize(nodenum);

    long node_count = 0;
    long edge_count = 0;
    while (getline(infile, line))
    {
        //        getline(infile, line);
        stringstream ss(line);
        ss >> fid;
        if (strcmp(fid.c_str(), "#") == 0) continue;
        ss >> tid;
        long from = atol(fid.c_str());
        long to = atol(tid.c_str());

        if (node_dict.find(from) == node_dict.end()) {
            node_dict[from] = node_count; node_count++;
        }
        if (node_dict.find(to) == node_dict.end()) {
            node_dict[to] = node_count; node_count++;
        }
        //create the in-neighbor adj list
        G_IN[node_dict[to]].push_back(node_dict[from]);
        //create the out-neighbor adj list
        G_OUT[node_dict[from]].push_back(node_dict[to]);
        // for undirected graph
        G_IN[node_dict[from]].push_back(node_dict[to]);
        G_OUT[node_dict[to]].push_back(node_dict[from]);
        edge_count = edge_count + 2;
    }
    //sort the in and out Adj list
    for (vector<Adjlist>::iterator it = G_IN.begin(); it != G_IN.end(); it++)
        sort(it->begin(), it->end());
    for (vector<Adjlist>::iterator it = G_OUT.begin(); it != G_OUT.end(); it++)
        sort(it->begin(), it->end());
    for (int i = 0; i < nodenum; i++)
        degree[i] = (int)G_IN[i].size();
    cout << "the number of node:" << node_count << endl;
    cout << "the number of edge:" << edge_count << endl;
    cout << "# nodes have in-neigs: " << G_IN.size() << endl;
    cout << "# nodes have out-neigs: " << G_OUT.size() << endl;

    //for (auto it = G_IN[0].begin(); it != G_IN[0].end(); it++)
    //    cout << *it << " ";

    infile.close();
}

vector <int> Graph::RemoveNode(int k, NodeID id) {
    /*ɾ���ڵ�id��������follower��������Ҳ����*/
    vector <int> remove_node;
    vector <int> deg(degree);
    remove_node.push_back(id);
    queue <int> Q;
    Q.push(id);
    while (!Q.empty()) {
        int v = Q.front();
        Q.pop();
        if (v != id)
            remove_node.push_back(v);
        for (Adjlist::iterator it = G_IN[v].begin(); it != G_IN[v].end(); it++) {
            deg[*it] --;
            if (deg[*it] == k - 1)
                Q.push(*it);
        }
    }
    sort(remove_node.begin(), remove_node.end());
    return remove_node;
}

int Graph::GetFollowerNum(int k, vector<NodeID>idx) {
    int cnt = 0;
    vector <int> deg(degree);
    queue <int> Q;
    for (int i = 0; i < idx.size(); i++) {
        Q.push(idx[i]);
    }

    while (!Q.empty()) {
        int v = Q.front();
        Q.pop();
        int n = count(idx.begin(), idx.end(), v);
        if (n == 0)        //���vû����idx�У���v��һ��follower_exact���������
            follower_exact.push_back(v);
        cnt++;
        deg[v] = 0;
        for (Adjlist::iterator it = G_IN[v].begin(); it != G_IN[v].end(); it++) {
            deg[*it] --;
            if (deg[*it] == k - 1)
                Q.push(*it);
        }
    }

    return cnt;
}

vector <int> Graph::Get_follower_exact() {
    return follower_exact;
}

vector <int> Graph::FindBest(int k, vector <int> deg) {
    /*���ص�ǰkcore�У�follower�������Ľڵ��ID��follower����*/
    NodeID maxID = 0;//�洢��ǰ��ѽڵ��id
    int maxflnum = 0;//�洢��ǰ��ѽڵ��follower����

    for (int i = 0; i < nodenum; i++) {
        if (deg[i] <= 0) {//���i�Ķ�<=0��˵��i����һ�ֵ������Ѿ���ɾ�ˣ����־Ͳ�������
            continue;
        }
        vector <int>d(deg);
        queue<int> tmp_remove;
        tmp_remove.push(i);
        int cnt_flnum = 0;

        //cout << endl << "cnt_flnum:";
        while (!tmp_remove.empty()) {
            int v = tmp_remove.front();
            //cout << v << " ";
            tmp_remove.pop();
            d[v] = 0;
            cnt_flnum++;
            for (Adjlist::iterator it = G_IN[v].begin(); it != G_IN[v].end(); it++) {
                d[*it] --;
                if (d[*it] == k - 1) {
                    tmp_remove.push(*it);
                    //cnt_flnum++;
                }
            }
        }

        //cout << cnt_flnum << " ";

        if (cnt_flnum > maxflnum) {
            maxID = i;
            maxflnum = cnt_flnum;
        }
    }

    //cout << "maxID:" << maxID << endl;
    //cout << "maxflnum:" << maxflnum << endl;
    vector <int> rslt;
    rslt.push_back(maxID);
    rslt.push_back(maxflnum);
    return rslt;
}

int Graph::Greedy(int k, int b) {
    /*̰���㷨������b���ڵ��follower����*/
    vector <int> deg(degree);//Ϊ�˲��ı�ԭͼ��degree����¡һ��Ϊdeg
    for (int i = 0; i < b; i++) {
        /*cout << "----" << i << "---:";
        for (int j = 0; j < deg.size(); j++)
            cout <<"��"<<j<<":" << deg[j] << " ";
        cout << endl;
        cout << "-----------------------------" << endl;*/
        vector <int> currtmax = FindBest(k, deg);
        greedy_rslt_ID.push_back(currtmax[0]);
        greedy_rslt_flnum += currtmax[1];

        queue <int> Q;
        Q.push(currtmax[0]);
        while (!Q.empty()) {
            int v = Q.front();
            Q.pop();
            deg[v] = 0;
            for (Adjlist::iterator it = G_IN[v].begin(); it != G_IN[v].end(); it++) {
                deg[*it] --;
                if (deg[*it] == k - 1)
                    Q.push(*it);
            }
        }
    }
    return greedy_rslt_flnum;
}

void Graph::Compute_un_dominated(int k) {
    vector <NodeID> follower_set;       //�洢������Ҫ����֦���Ľڵ�
    for (int i = 0; i < degree.size(); i++) {
        vector <int> deg(degree);
        queue <int> Q;
        Q.push(i);

        while (!Q.empty()) {
            int v = Q.front();
            Q.pop();
            if (v != i)        //���vû����idx�У���v��һ��follower_exact���������
                follower_set.push_back(v);
            deg[v] = 0;
            for (Adjlist::iterator it = G_IN[v].begin(); it != G_IN[v].end(); it++) {
                deg[*it] --;
                if (deg[*it] == k - 1)
                    Q.push(*it);
            }
        }

    }
    for (int i = 0; i < degree.size(); i++) {
        int n = count(follower_set.begin(), follower_set.end(), i);
        if (n == 0)
            un_dominated.push_back(i);
    }
}

vector <int> Graph::Get_un_dominated(int k) {
    Compute_un_dominated(k);
    return  un_dominated;
}

// PYBIND11_MODULE(kcore, m) {
//     pybind11::class_<Graph>(m, "Graph")
//         .def(pybind11::init())
//         .def("loadUndirGraph", &Graph::loadUndirGraph)
//         .def("RemoveNode", &Graph::RemoveNode)
//         .def("GetFollowerNum", &Graph::GetFollowerNum)
//         .def("Get_follower_exact", &Graph::Get_follower_exact)
//         .def("FindBest", &Graph::FindBest)
//         .def("Greedy", &Graph::Greedy)
//         .def("Compute_un_dominated", &Graph::Compute_un_dominated)
//         .def("Get_un_dominated", &Graph::Get_un_dominated);

// }
