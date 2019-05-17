#!/usr/bin/env python3

import vecto.embeddings
import vecto.benchmarks.analogy.analogy
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys



def main(poincare=None, w2v300=None):
    poincare_sorted = sorted(poincare, key=lambda x: (x['experiment_setup']['category'] + ':' + x['experiment_setup']['subcategory']))
    w2v300_sorted = sorted(w2v300, key=lambda x: (x['experiment_setup']['category'] + ':' + x['experiment_setup']['subcategory']))
    for t in range(len(poincare)):
        p = poincare_sorted[t]
        w = w2v300_sorted[t]

        print('category = %s' % p['experiment_setup']['category'])
        print('subcategory = %s' % p['experiment_setup']['subcategory'])
        acc_p = p['result']['accuracy']
        acc_w = w['result']['accuracy']
        plt.barh(t * 2, acc_p, label=p['experiment_setup']['subcategory'], color="#1E7F00")
        plt.barh(t * 2 + 1, acc_w, label=p['experiment_setup']['subcategory'], color="#FF5B70")

    plt.legend(['poincare', 'w2v300'])
    plt.savefig("results2.pdf", bbox_inches="tight")

    print('poincare: len=%d' % len(poincare))
    # print(poincare[0])
    print('details: %s' % poincare[0]['details'][0])
    print('result: %s' % poincare[0]['result'])
    print('exp: %s' % poincare[0]['experiment_setup'])
    for k, v in poincare[0].items():
        print('%s: %s' % (str(k), str(type(v))))



if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as pf:
        poincare = pickle.load(pf)

    with open(sys.argv[2], 'rb') as pf2:
        w2v300 = pickle.load(pf2)

    main(poincare=poincare, w2v300=w2v300)
