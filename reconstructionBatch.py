#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hype.graph import eval_reconstruction, load_adjacency_matrix
import argparse
import numpy as np
import torch
import os
import sys
import timeit
from hype.lorentz import LorentzManifold
from hype.euclidean import EuclideanManifold, TranseManifold
from hype.poincare import PoincareManifold
from hype.checkpoint import LocalCheckpoint


MANIFOLDS = {
    'lorentz': LorentzManifold,
    'euclidean': EuclideanManifold,
    'transe': TranseManifold,
    'poincare': PoincareManifold
}

np.random.seed(42)


def calc_rank_map(fpath='', adj=None):
    chkpnt = torch.load(fpath)
    manifold = MANIFOLDS[chkpnt['conf']['manifold']]()

    lt = chkpnt['embeddings']

    if not isinstance(lt, torch.Tensor):
        lt = torch.from_numpy(lt).cuda()

    tstart = timeit.default_timer()
    meanrank, maprank = eval_reconstruction(adj, lt, manifold.distance,
                                            workers=2, progress=False)
    etime = timeit.default_timer() - tstart

    print(f'Mean rank: {meanrank}, mAP rank: {maprank}, time: {etime}')

    return (chkpnt, manifold, lt, meanrank, maprank, etime)

def save_snapshot(chkpnt=None,
                  fpath='',
                  manifold=None,
                  lt=None,
                  elapsed=0,
                  loss=0,
                  meanrank=0.0,
                  maprank=0.0):
    sqnorms = manifold.pnorm(lt)
    lmsg = {
        'epoch': chkpnt['epoch'],
        'elapsed': elapsed,
        'loss': loss,
        'sqnorm_min': sqnorms.min().item(),
        'sqnorm_avg': sqnorms.mean().item(),
        'sqnorm_max': sqnorms.max().item(),
        'mean_rank': meanrank,
        'map_rank': maprank
    }
    print('save lmsg to %s.' % fpath)
    checkpoint = LocalCheckpoint(
        fpath,
        include_in_all={'conf' : chkpnt['conf'], 'objects' : chkpnt['objects']},
        start_fresh=False
    )
    chk_obj = checkpoint.initialize(None)

    checkpoint.save({
        'conf': chkpnt['conf'],
        'model': chkpnt['model'],
        'embeddings': lt,
        'epoch': chkpnt['epoch'],
        'manifold': chkpnt['manifold'],
        'lmsg': lmsg
    })

    return True


def calc_adj(data=None):
    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    return adj


def main(embdir='./emb150/', iterations=300, dataset='./wordnet/noun_closure.csv'):
    # set default tensor type
    torch.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = torch.device('cuda:0')

    format = 'hdf5' if dataset.endswith('.h5') else 'csv'
    dset = load_adjacency_matrix(dataset, format=format)

    sample_size = len(dset['ids'])
    sample = np.random.choice(
        len(dset['ids']), size=sample_size, replace=False)

    adj = {}
    print('calc adj...')
    for i in sample:
        end = dset['offsets'][i + 1] if i + 1 < len(dset['offsets']) \
            else len(dset['neighbors'])
        adj[i] = set(dset['neighbors'][dset['offsets'][i]:end])
    print('calc done...')

    manifolds = ['euclidean',
                 # 'transe',
                 'poincare',
                 'lorentz']
    dimensions = [5, 10, 20, 50, 100, 200]

    for mani in manifolds:
        for dim in dimensions:
            key_json = '%s%d' % (mani, dim)
            for cp in range(9, iterations, 10):
                infile = 'emb_%s_%d.bin.%d' % (mani, dim, cp)
                fpath = os.path.join(embdir, infile)
                print('load %s...' % fpath)
                if os.path.exists(fpath):
                    chkpnt, manifold, lt, meanrank, maprank, elapsed = calc_rank_map(
                        fpath=fpath, adj=adj)
                    ret = save_snapshot(chkpnt=chkpnt, fpath=fpath, manifold=manifold,
                                        lt=lt, elapsed=elapsed, loss=0, meanrank=meanrank,
                                        maprank=maprank)


if __name__ == '__main__':
    main(embdir=sys.argv[1])
