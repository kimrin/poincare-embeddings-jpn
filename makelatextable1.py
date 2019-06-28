#!/usr/bin/env python3

import pickle
import numpy as np
import os
import sys
import csv
import json


PRE_STRING = r"""
\begin{table}[htb]
  \begin{tabular}{|l|c||r|r|r|r|r|r|} \hline"""

POST_STRING = r"""  \end{tabular}
\end{table}
"""


def main(jsonfile=''):
    with open(jsonfile, 'r') as fp:
        js = json.load(fp)

    manifolds = ['euclidean',
                 # 'transe',
                 'poincare',
                 'lorentz']
    dimensions = [5, 10, 20, 50, 100, 200]

    print(PRE_STRING)
    print('  manifold & RANK/MAP & ' +
          ' & '.join(list(map(str, dimensions))) + r' \\ \hline')
    for mani in manifolds:
        pre1 = '  %s & RANK & ' % mani
        pre2 = ' & MAP & '
        rank = []
        maps = []
        for dim in dimensions:
            key_json = '%s%d' % (mani, dim)
            con = js[key_json]
            rank.append('%4.4f' % con["mean_rank"])
            maps.append('%4.4f' % con["MAP"])
        pre1 += ' & '.join(rank) + ' \\\\ \\cline{3-8}'
        pre2 += ' & '.join(maps) + ' \\\\ \\hline'
        print(pre1)
        print(pre2)

    print(POST_STRING)


if __name__ == '__main__':
    main(jsonfile=sys.argv[1])
