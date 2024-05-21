#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

# import random
import numpy as np
from argparse import ArgumentParser

from read_data import my_prepare_input
from models.fake_flow import FakeFlow

# # Reproducibility
# np.random.seed(0)
# random.seed(0)

if __name__ == '__main__':
    parser = ArgumentParser()

    # General parameters:
    parser.add_argument("-d", "--dataset", help="Dataset name", default='fakedes')
    parser.add_argument("-sn", "--segments_number", help="Number of segments: the default value is 10", default=10, type=int)
    parser.add_argument("-of", "--overwrite_features", help="overwrite features", default=0, type=int)
    parser.add_argument("-o", "--output_dir", help="output directory where to save the model", required=True, type=str)
    parser.add_argument("-m", "--mode", help="train, test, or apply", default='train')
    parser.add_argument("-ub", "--use_branches", help="both_branches, affective_branch or topic_branch", default='both_branches')
    parser.add_argument("-ep", "--embedding_path", help="embedding path", default='./my_embeddings.vec', type=str)
    parser.add_argument("-es", "--embedding_size", help="embedding size", default=300, type=int)

    # Network parameters:
    parser.add_argument("-rs", "--rnn_size", help="size rnn", default=8, type=int)
    parser.add_argument("-nf", "--num_filters", help="number of filters", default=16, type=int)
    parser.add_argument("-fs", "--filter_sizes", help="filter sizes", nargs='+', default=[2,3,4], type=int)
    parser.add_argument("-ar", "--activation_rnn", help="activation rnn", default="tanh", type=str)
    parser.add_argument("-ps", "--pool_size", help="pool size", default=3, type=int)
    parser.add_argument("-ac", "--activation_cnn", help="activation_cnn", default="relu", type=str)
    parser.add_argument("-d1", "--dense_1", help="dense 1", default=8, type=int)
    parser.add_argument("-aa", "--activation_attention", help="activation attention", default="softmax", type=str)
    parser.add_argument("-d2", "--dense_2", help="dense 2", default=8, type=int)
    parser.add_argument("-d3", "--dense_3", help="dense 3", default=8, type=int)
    parser.add_argument("-dt", "--dropout", help="dropout", default=0.3910, type=float)
    parser.add_argument("-ml", "--max_senten_len", help="max sentence length", default=500, type=int)
    parser.add_argument("-vb", "--vocab", help="vocab", default=1000000, type=int)
    parser.add_argument("-me", "--max_epoch", help="max epoch", default=50, type=int)
    parser.add_argument("-bs", "--batch_size", help="batch size", default=16, type=int)

    args = parser.parse_args()
    args.filter_sizes = tuple(args.filter_sizes)

    # -- reading data and pre-procesing
    train, dev, test = my_prepare_input(dataset=args.dataset, segments_number=args.segments_number, text_segments=True, overwrite=args.overwrite_features)

    # -- building FakeFlow model
    EF = FakeFlow(vars(args))

    # -- prepare word embeddings
    print("\n----> BEFORE prepare_input:", train['features'].shape, '\n')
    EF.prepare_fake_flow_input(train, dev, test)

    EF.run_model(args.output_dir, type_ = args.mode, use_branches=args.use_branches)