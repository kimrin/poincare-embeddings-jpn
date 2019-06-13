#!/usr/bin/env python3

import pickle
import numpy as np
import os
import sys
import csv
import json
import torch


def main(embdir='./emb150/'):
    # manifolds = ['euclidean',
    #              # 'transe',
    #              'poincare',
    #              'lorentz']
    # dimensions = [5, 10, 20, 50, 100, 200]
    manifolds = ['poincare',
                 'lorentz']
    dimensions = [5, 10, 20]

    for mani in manifolds:
        for dim in dimensions:
            key_json = '%s%d' % (mani, dim)
            with open(os.path.join(embdir, '%s.conv.csv' % key_json), 'w') as csvf:
                csvwriter = csv.DictWriter(csvf, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, fieldnames=[
                                           'iterations', 'mean_rank', 'mean_map'])
                csvwriter.writeheader()
                for cp in range(9, 150, 10):
                    infile = 'emb_%s_%d.bin.%d' % (mani, dim, cp)
                    fpath = os.path.join(embdir, infile)
                    if os.path.exists(fpath):
                        obj = torch.load(fpath)
                        # print(obj)
                        for k, v in obj.items():
                            print('key=%s: value = %s' % (k, str(type(v))))
                        # print(obj['conf'])
                        if obj.get('lmsg', None) is not None:
                            for k, v in obj['lmsg'].items():
                                print(k, ':', v)

if __name__ == '__main__':
    main(embdir=sys.argv[1])