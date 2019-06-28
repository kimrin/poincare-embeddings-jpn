#!/usr/bin/env python3

import pickle
import os
import sys
import csv
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def makegraph(embdir='./emb150/'):
    manifolds = ['euclidean',
                 # 'transe',
                 'poincare',
                 'lorentz']
    dimensions = [5, 10, 20, 50, 100, 200]

    for mani in manifolds:
        for dim in dimensions:
            key_json = '%s%d' % (mani, dim)
            with open(os.path.join(embdir, '%s.conv.csv' % key_json), 'w') as csvf:
                csvwriter = csv.DictWriter(csvf, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, fieldnames=[
                                           'iterations', 'mean_rank', 'map_rank'])
                csvwriter.writeheader()
                for cp in range(9, 300, 10):
                    infile = 'emb_%s_%d.bin.%d' % (mani, dim, cp)
                    fpath = os.path.join(embdir, infile)
                    if os.path.exists(fpath):
                        obj = torch.load(fpath)
                        if obj.get('lmsg', None) is not None:
                            # for k, v in obj['lmsg'].items():
                            #     print(k, ':', v)
                            csvwriter.writerow(
                                {'iterations': cp, 'mean_rank':  obj['lmsg']['mean_rank'], 'map_rank': obj['lmsg']['map_rank']})


def plotgraph(embdir='./emb150/', pdffile='poincare_map_and_mean_rank.'):
    manifolds = ['euclidean',
                 # 'transe',
                 'poincare',
                 'lorentz']
    dimensions = [5, 10, 20, 50, 100, 200]

    ref_rank = {'euclidean': [3542.3, 2286.9, 1685.9, 1281.7, 1187.3, 1157.3],
                'poincare': [4.9, 4.02, 3.84, 3.98, 3.9, 3.83]}
    ref_map = {'euclidean': [0.024, 0.059, 0.087, 0.140, 0.162, 0.168],
               'poincare': [0.823, 0.851, 0.855, 0.86, 0.857, 0.87]}

    rows = len(manifolds)
    columns = len(dimensions)
    # fieldnames = [
    #    'iterations', 'mean_rank', 'map_rank']

    # set path
    pp = PdfPages(pdffile + 'pdf')
    for rank in ['mean_rank', 'map_rank']:
        for row, mani in enumerate(manifolds):
            for column, dim in enumerate(dimensions):
                fig = plt.figure()
                key_json = '%s%d' % (mani, dim)
                datapath = os.path.join(embdir, '%s.conv.csv' % key_json)
                df = pd.read_csv(datapath)
                # plt.subplot(rows, columns, (row * columns + column) + 1)
                if rank == 'mean_rank':
                    color = 'b'
                    # if mani == 'euclidean':
                    #     plt.hlines(
                    #         y=ref_rank[mani][column], xmin=0, xmax=300, colors='b', linestyles='dashed')
                    # else:
                    #     plt.hlines(
                    #         y=ref_rank['poincare'][column], xmin=0, xmax=300, colors='b', linestyles='dashed')
                else:
                    color = 'g'
                    # if mani == 'euclidean':
                    #     plt.hlines(y=ref_map[mani][column], xmin=0,
                    #                xmax=300, colors='g', linestyles='dashed')
                    # else:
                    #     plt.hlines(
                    #         y=ref_map['poincare'][column], xmin=0, xmax=300, colors='g', linestyles='dashed')
                plt.title("%s manifold: dim = %d" % (mani, dim))
                plt.plot(df['iterations'], df[rank], color, label=rank)
                plt.legend()

                # save figure
                pp.savefig(fig)

    # close file
    pp.close()


if __name__ == '__main__':
    # makegraph(embdir=sys.argv[1])
    plotgraph(embdir=sys.argv[1])
