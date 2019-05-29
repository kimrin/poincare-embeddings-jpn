#!/usr/bin/env python3

import pickle
import numpy as np
import os
import sys
import csv
import json
from gensim.models.poincare import PoincareKeyedVectors
from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.models.poincare import ReconstructionEvaluation
from gensim.test.utils import datapath


def generate_data(filename='./wordnet/noun_closure.csv'):
    outfilename = os.path.splitext(filename)[0] + '.tsv'
    with open(filename, 'r', encoding='utf-8') as fp:
        with open(outfilename, 'w', encoding='utf-8') as outp:
            reader = csv.DictReader(fp)
            writer = csv.DictWriter(
                outp, delimiter='\t', quoting=csv.QUOTE_NONE, fieldnames=['id1', 'id2'])
            # writer.writeheader()
            for r in reader:
                writer.writerow(
                    {'id1': '%s' % r['id1'], 'id2': '%s' % r['id2']})

    return outfilename


def main(outfile=''):
    manifolds = ['euclidean',
                 # 'transe',
                 'poincare',
                 'lorentz']
    dimensions = [5, 10, 20, 50, 100, 200]
    js = {}
    
    for mani in manifolds:
        for dim in dimensions:
            key_json = '%s%d' % (mani, dim)            
            txtdir = './emb150txt/%s%d/' % (mani, dim)
            txtfile = txtdir + ('%s%d.txt' % (mani, dim))
            print('loading %s...' % txtfile)
            keyvalues = PoincareKeyedVectors.load_word2vec_format(txtfile,
                                                                  fvocab=None,
                                                                  binary=False,
                                                                  encoding='utf8',
                                                                  unicode_errors='strict',
                                                                  limit=None,
                                                                  datatype=np.float64)

            pr = generate_data()
            actual_dim = dim
            if mani == 'lorentz':
                actual_dim += 1
            eva = ReconstructionEvaluation(pr, keyvalues)
            # print('filename=%s: ' % txtfile)
            res = eva.evaluate()
            js[key_json] = res
    with open(outfile, 'w') as fp:
        json.dump(js,fp)

if __name__ == '__main__':
    main(sys.argv[1])
