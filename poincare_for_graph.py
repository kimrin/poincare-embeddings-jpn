#!/usr/bin/env python3

import pickle
import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plotgraph(csvfile='po1500.csv'):
    ref_rank = {'euclidean': [3542.3, 2286.9, 1685.9, 1281.7, 1187.3, 1157.3],
                'poincare': [4.9, 4.02, 3.84, 3.98, 3.9, 3.83]}
    ref_map = {'euclidean': [0.024, 0.059, 0.087, 0.140, 0.162, 0.168],
               'poincare': [0.823, 0.851, 0.855, 0.86, 0.857, 0.87]}

    prefix, suffix = os.path.splitext(csvfile)
    outfile = prefix + '.pdf'

    pp = PdfPages(outfile)
    for rank in ['loss', 'mean_rank', 'map_rank']:
        fig = plt.figure()
        datapath = csvfile
        df = pd.read_csv(datapath)
        if rank == 'mean_rank':
            color = 'b'
        elif rank == 'loss':
            color = 'm'
        else:
            color = 'g'
        plt.title("%s of poincare manifold: dim = 10" % rank)
        plt.plot(df['epoch'], df[rank], color, label=rank)
        plt.legend()

        # save figure
        pp.savefig(fig)

    # close file
    pp.close()


if __name__ == '__main__':
    plotgraph(csvfile=sys.argv[1])
