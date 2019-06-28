#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import sys
import timeit
from pprint import pprint

conf_shell = {"checkpoint": "nouns.bin", "dset": "wordnet/noun_closure.csv", "dim": 10, "manifold": "poincare", "lr": 1.0, "epochs": 1500, "batchsize": 50, "negs": 50, "burnin": 20, "dampening": 1.0, "ndproc": 4, "eval_each": 100, "fresh": True, "debug": False, "gpu": 0, "sym": False, "maxnorm": 500000, "sparse": True, "burnin_multiplier": 0.01, "neg_multiplier": 0.1, "quiet": False, "lr_type": "constant", "train_threads": 1}

def get_config(filepath=''):
    chkpnt = torch.load(filepath)
    conf = chkpnt['conf']
    print('config:')
    pprint(conf)

if __name__ == '__main__':
    get_config(filepath=sys.argv[1])
    pprint(conf_shell)
