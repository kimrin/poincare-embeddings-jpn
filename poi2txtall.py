#!/usr/bin/env python3

import os
import sys
import subprocess
from poincare2txt import main as poi2txt


def mainroutine():
    manifolds = ['euclidean',
                 # 'transe',
                 'poincare',
                 'lorentz']
    dimensions = [5, 10, 20, 50, 100, 200]
    for mani in manifolds:
        for dim in dimensions:
            filename = './emb150/emb_%s_%d.bin' % (mani, dim)
            txtdir = './emb150txt/%s%d/' % (mani, dim)
            os.makedirs(txtdir)
            txtfile = txtdir + ('%s%d.txt' % (mani, dim))
            poi2txt(filename, os.path.splitext(txtfile))


if __name__ == '__main__':
    mainroutine()
