#ifndef __STREAMPPR_H
#define __STREAMPPR_H

#include <cblas.h>
#include <iostream>
#include <queue>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

typedef unsigned long long uLL;
typedef long long LL;

class StreamPPR{
private:
    vector< vector<int64_t> > edge_;
    vector<int> in_deg_, out_deg_;
    int64_t num_node_, num_edge_, dim_;
    np::ndarray r_, x_, z_;
    float alpha_, epsilon_;
    queue<int64_t> q;
    vector<bool> inq_;

    float *delta_rv_;

private:
    void check_push(int64_t u, float *r);

public:
    /*
    edge_list: 2 * num_edge, numpy ndarray. Node id starting from 0.
    X:         num_node * embedding dimension
    */
    StreamPPR(np::ndarray x, np::ndarray edge_index, float alpha, float epsilon);

    ~StreamPPR();

    void basic_propagation();

    void add_edge(int64_t u, int64_t v);

    void add_node_signal(int64_t u, np::ndarray delta_ru);

    void rotate_space(np::ndarray B);

    np::ndarray get_embedding();

    np::ndarray get_residual();

    np::ndarray get_signal();
};

const char* init();

#endif
