# DAMF

Here is the code for the paper "Accelerating Dynamic Network Embedding with Billions of Parameter Updates to Milliseconds [[arXiv](https://arxiv.org/abs/2306.08967)]" submitted to **KDD2023** . In this code repository we provide:

- **DAMF**: A runnable code for DAMF. This includes,  the full version of DAMF (`DAMF.py`), which requires a more complex setup process, and a basic version without the PPR enhancements (`DAMF_unenhanced.py`).

- **Datasets**: Some of the datasets used for the experiments in our paper. As the datasets used in our experiments are large, due to the space limit, we have only included a small part of datasets.

- **Evaluator**: Code for generating training and test data, as well as code for measuring node classification, link prediction and graph reconstruction tasks.

- **Baseline replicated by us**: As we did not find the code for LocalAction, we replicated it ourselves in python, which will also be made public here.

### Setup

We are using Python version **3.9.12** and install the dependency packages with the following command:

```
pip install -r requirements.txt
```

If you do not require dynamic embedding enhancements, you only need to install the following Python dependency packages. Otherwise, you will need to install Boost and compile the dynamic embedding enhancement module. To install the dynamic embedding enhancement module, you may need to make sure that **C++**, **cmake** and **BLAS** (e.g. OpenBLAS) are already installed. 

Install Boost and BLAS:

```
sudo apt-get update
sudo apt-get install libboost-all-dev
sudo apt-get install cblas
```

Compile the dynamic embedding enhancement module:

```
mkdir build
cd build
cmake ..
make
cd ..
```

### Input

The input should be a sparse matrix in `.mat` format. You can refer to the corresponding functions in `utils.py` for details.

### Demo

If everything is ready, try the following demo, which adds the points of the wiki dataset in turn to a graph initialized with 1000 nodes and computes and updates the 128-dimenspional embedding. The generated results will be exported to `embds/wiki`. Then, you can call  `nc.py` to evaluate the unenhanced version of DAMF (a.k.a. `damf1` in the algo_list) at the Node Classification task.

```
python DAMF_unenhanced.py --data wiki
cd eval
python nc.py --data wiki --algo damf1
```

### Embedding


```
python DAMF_unenhanced.py --data [data name] --type ["train", "full"]
```

You can use `--d` to adjust the embedding dimenspional (default is 128) and `--init` to adjust the initial number of nodes (default is 1000).

```
python DAMF.py --data [data name] --type ["train", "full"]
```

You can also use `--epsilon` and `--alpha` to adjust the hyperparameters for PPR.



### Evaluate

After ensuring that your embedding results have been saved in `embds/[algo]`, you can start the Evaluation process.

```
cd eval
```

##### Node Classification

```
python nc.py --data [data name] --algo [damf, damf1]
```

##### Link prediction

```
python lp.py --data [data name] --algo [damf, damf1]
```

##### Graph reconstruction

```
python gr.py --data [data name] --algo [damf, damf1]
```

### Baseline (LocalAction)

```
python LocalAction.py --data [data name] --type ["train", "full"]
```

```
python nc.py --data [data name] --algo la
```

In addition, if you want to evaluate other models, you need to add the model names to `eval/algo_list.py` first.
