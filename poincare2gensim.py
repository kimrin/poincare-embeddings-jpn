#!/usr/bin/env python3
import datetime
import os
import uuid
import logging
import progressbar
import sklearn
import numpy as np
import pickle
import sys

from gensim.models.poincare import PoincareKeyedVectors


def main(poincare=''):
    from gensim.models.poincare import PoincareModel
    pm = PoincareModel([], size=300, dtype=np.float64)
    emb = PoincareKeyedVectors.load_word2vec_format(poincare, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict', limit=None, datatype=np.float64)
    pm.kv = emb

    pm.save('w2v_poincare.pickle', pickle_protocol=4)
    pm2 = PoincareModel.load('w2v_poincare.pickle')

    pm2.train(10000, batch_size=10, print_every=1, check_gradients_every=None)
    pm2.save('w2v_poincare_after_train.pickle', pickle_protocol=4)

    # words = emb.vocab.keys()
    # for w in words:
    #     print('%s: %s' % (w, emb.ancestors(w)))

if __name__ == '__main__':
    main(poincare=sys.argv[1])
