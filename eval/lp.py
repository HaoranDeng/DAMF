#########################################################################
# File Name: eval_linkpred.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Tue 13 Nov 2018 01:56:10 PM +08
#########################################################################
#!/usr/bin/env/ python

import linkpred_util as lutil
import graph_util as gutil
import multiprocessing
import argparse
import os
import os.path

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

PARALLEL_LEVEL = multiprocessing.cpu_count()

print("CPU count: %d"%PARALLEL_LEVEL)


parser = argparse.ArgumentParser(description='Process...')
parser.add_argument('--algo', type=str, help='algorithm name')
parser.add_argument('--d', type=int, default=128, help='embedding dimensionality')
parser.add_argument('--data', type=str, help='graph dataset name')
# parser.add_argument('--ratio', type=float, default=0.3, help='test data ratio')
args = parser.parse_args()

lpeval = lutil.LinkPredictionEval(args.data, args.algo, args.d)
#lpeval.eval4()
lpeval.eval()
