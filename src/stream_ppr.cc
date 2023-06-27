#include "stream_ppr.h"	

// beta = 0

/*************************************************
Function:       init
Description:    Before calling this library, you should call this initialization function to initialize
                the python settings in the boost library, otherwise an error may be caused.
*************************************************/
const char* init() {
    Py_Initialize();
    np::initialize();
    return "Init.";
}


/*************************************************
Function:       Graph
Description:    Constructor for Graph.
Input:          edge_index_np: A 2-d numpy.ndarray shapes like [2, num_edge_] indicates edge index for the original graph.
                star_np: A 1-d bool numpy.ndarray marks the important nodes.
*************************************************/
StreamPPR::StreamPPR(np::ndarray x_np, np::ndarray edge_index_np, float alpha, float epsilon):
    r_(x_np.copy()),
    x_(x_np.copy()),
    z_(np::zeros(p::make_tuple(x_np.shape(0), x_np.shape(1)), x_np.get_dtype())),
    alpha_(alpha),
    epsilon_(epsilon) {

    // Check "edge_index"
    int64_t nd_edge_index = edge_index_np.get_nd();
    if (nd_edge_index != 2 || edge_index_np.shape(0) != 2)
        throw std::runtime_error("\"edge_index\" must be 2-dimensional numpy.ndarray shapes like [2, num_edge]. ");
    if (edge_index_np.get_dtype() != np::dtype::get_builtin<int64_t>())
        throw std::runtime_error("\"edge_index\" must be int64 numpy.ndarray. ");

    // check "X"
    int64_t nd_x = x_np.get_nd();
    if (nd_x != 2)
        throw std::runtime_error("\"X\" must be 2-dimensional numpy.ndarray shapes like [num_node, embedding_dimension]. ");
    if (x_np.get_dtype() != np::dtype::get_builtin<float>())
        throw std::runtime_error("\"X\" must be float32. ");
    
    num_node_ = x_np.shape(0);
    dim_ = x_np.shape(1);

    num_edge_ = edge_index_np.shape(1);
    int64_t *edge_index = reinterpret_cast<int64_t *>(edge_index_np.get_data());
    for (int64_t i=0;i<2*num_edge_;i++) {
        if (edge_index[i] < 0) {
            throw std::runtime_error("Negative node_id in \"edge_index\". ");
        }
        if (edge_index[i] >= num_node_) {
            throw std::runtime_error("edge_index[i] > num_node");
        }
    }

    edge_.resize(num_node_);
    in_deg_.resize(num_node_);
    out_deg_.resize(num_node_);

    for (int i=0;i<num_node_;i++) {
        in_deg_.push_back(0);
        out_deg_.push_back(0);
        inq_.push_back(false);
    }

    for (LL i=0;i<num_edge_;i++) {
        int u = edge_index[i];
        int v = edge_index[num_edge_+i];
        if (u == v) continue; 
        edge_[u].push_back(v);
        out_deg_[u] += 1;
        in_deg_[v] += 1;
    }


    float *r = reinterpret_cast<float *>(r_.get_data());
    for (int i=0;i<num_node_;i++) check_push(i, r);

    delta_rv_ = new float[dim_];

    // Output on screen
    std::cerr << "::: Graph(#Node: " << num_node_ << ", #Edge: " << num_edge_ << ")." << std::endl;
    std::cerr << "::: epsilon: " << epsilon_ << std::endl;
}

StreamPPR::~StreamPPR() {
    delete delta_rv_;
}

inline void StreamPPR::check_push(int64_t u, float *r) {
    // float *r = reinterpret_cast<float *>(r_.get_data());
    if (inq_[u] == false) {
        for (int i=u*dim_;i<u*dim_+dim_;i++) {
            if (abs(r[i]) > epsilon_) {
                q.push(u);
                inq_[u] = true;
                break;
            }
        }
    }
    // float res = cblas_snrm2(dim_, r+u*dim_, 1);
    // //if (inq_[u] == false && res > epsilon_ * out_deg_[u]) {
    // if (inq_[u] == false && res > epsilon_ ) {
    //     q.push(u);
    //     inq_[u] = true;
    // }
    return ;   
}

void StreamPPR::basic_propagation() {
    // time_t start_t, end_t;
    // start_t = clock();
    float *r = reinterpret_cast<float *>(r_.get_data());
    int cnt = 0;
    float *tmp = new float[dim_];
    while (!q.empty()) {
        int s = q.front(); q.pop();
        inq_[s] = false;
        z_[s] += alpha_ * r_[s];
        
        for (int i=0;i<dim_;i++) {
	    // assert(out_deg_[s] > 0);
            tmp[i] = (1-alpha_) / out_deg_[s] * r[s*dim_+i];
        }        
        for (int i=0;i<(int)edge_[s].size();i++) {
            int t = edge_[s][i];
            if (t == s) continue;
            cnt++;
            for (int d=0;d<dim_;d++) {
                r[t * dim_ + d] += tmp[d];
            }
            check_push(t, r);
        }
        for (int i=s*dim_;i<s*dim_+dim_;i++) r[i] = 0;
    }
    delete tmp;
    // end_t = clock();
    // std::cerr << "Total propagation: " << cnt << std::endl;
    // std::cerr << "Time" << (end_t-start_t) / 1000 << "ms" << std::endl;
    return ;
}

// adding a edge (u, v)
void StreamPPR::add_edge(int64_t u, int64_t v) {
    float *r = reinterpret_cast<float *>(r_.get_data());
    float *z = reinterpret_cast<float *>(z_.get_data());

    edge_[u].push_back(v);
    out_deg_[u]++;
    in_deg_[v]++;

    // Step 1: clear delta_rv_
    memset(delta_rv_, 0, sizeof(float) * dim_);
    
    // Step 2: r[v] += (1-alpha) * z[u] / ( alpha * deg_u )
    float factor = (1.0-alpha_) / (alpha_ * out_deg_[u]);
    cblas_saxpy(dim_, factor, z+u*dim_, 1, r+v*dim_, 1);    
    
    // Step 3: Enqueue the node with residual > epsilon * deg
    check_push(v, r);
     

    // Step 4: 
    for (int i=0;i<(int)edge_[u].size();i++) {
        int w = edge_[u][i];
        // factor <- (1-alpha)/alpha * ( 1/dt - 1/(dt-1) )
        factor = 1.0 / out_deg_[u];
        if (out_deg_[u] > 1) 
            factor = factor - 1.0 / (out_deg_[u]-1);
        factor *= (1.0-alpha_) / alpha_;
        cblas_saxpy(dim_, factor, z+u*dim_, 1, r+w*dim_, 1);
        check_push(w, r);
    }
    return ;
}

void StreamPPR::add_node_signal(int64_t u, np::ndarray delta_ru_np) {
    int64_t nd_delta_ru = delta_ru_np.get_nd();
    if (nd_delta_ru != 1 || delta_ru_np.shape(0) != dim_)
        throw std::runtime_error("\"delta_ru\" must be 1-dimensional numpy.ndarray with shape (dim, ). ");
    if (delta_ru_np.get_dtype() != np::dtype::get_builtin<float>())
        throw std::runtime_error("\"X\" must be float32. ");
    float *delta_ru = reinterpret_cast<float *>(delta_ru_np.get_data());
    float *r = reinterpret_cast<float *>(r_.get_data());
    float *x = reinterpret_cast<float *>(x_.get_data());

    cblas_saxpy(dim_, 1.0, delta_ru, 1, r+u*dim_, 1);
    cblas_saxpy(dim_, 1.0, delta_ru, 1, x+u*dim_, 1);

    check_push(u, r);
    return ;
}

void StreamPPR::rotate_space(np::ndarray B_np) {
    // Check "B"
    int64_t nd_B = B_np.get_nd();
    if (nd_B != 2 || B_np.shape(0) != dim_ || B_np.shape(1) != dim_)
        throw std::runtime_error("\"B\" must be 2-dimensional numpy.ndarray shapes like [dim_, dim_]. ");
    if (B_np.get_dtype() != np::dtype::get_builtin<float>())
        throw std::runtime_error("\"edge_index\" must be np.float32 numpy.ndarray. ");

    float *tmp = new float[num_node_ * dim_];

    float *r = reinterpret_cast<float *>(r_.get_data());
    float *z = reinterpret_cast<float *>(z_.get_data());
    float *x = reinterpret_cast<float *>(x_.get_data());
    float *B = reinterpret_cast<float *>(B_np.get_data());

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                num_node_, dim_, dim_, 1.0,                      // M, N, K, alpha
                r, dim_, B, dim_, 0.0, tmp, dim_);
    cblas_scopy(num_node_ * dim_, tmp, 1, r, 1);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                num_node_, dim_, dim_, 1.0,                      // M, N, K, alpha
                z, dim_, B, dim_, 0.0, tmp, dim_);
    cblas_scopy(num_node_ * dim_, tmp, 1, z, 1);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                num_node_, dim_, dim_, 1.0,                      // M, N, K, alpha
                x, dim_, B, dim_, 0.0, tmp, dim_);
    cblas_scopy(num_node_ * dim_, tmp, 1, x, 1);

    for (int i=0;i<num_node_;i++) check_push(i, r);

    return ;
}

np::ndarray StreamPPR::get_embedding() {
    return z_;
}

np::ndarray StreamPPR::get_residual() {
    return r_;
}

np::ndarray StreamPPR::get_signal() {
    return x_;
}

BOOST_PYTHON_MODULE(embedding_propagation)
{
    boost::python::def("init", init);
    boost::python::class_<StreamPPR>("StreamPPR", boost::python::init<np::ndarray, np::ndarray, float, float>())
        .def("basic_propagation", &StreamPPR::basic_propagation)
        .def("add_edge", &StreamPPR::add_edge)
        .def("add_node_signal", &StreamPPR::add_node_signal)
        .def("get_embedding", &StreamPPR::get_embedding)
        .def("get_residual", &StreamPPR::get_residual)
        .def("get_signal", &StreamPPR::get_signal)
        .def("rotate_space", &StreamPPR::rotate_space)
    ;
}


