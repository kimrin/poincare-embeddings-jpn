#!/usr/bin/env python3

import pickle
import logging
import argparse
import numpy as np
import torch
import os
import sys
import timeit


def main(binfile='', outfilelist=[]):
    obj = torch.load(binfile)
    for k, v in obj.items():
        print('key=%s: value = %s' % (k, str(type(v))))

    print(obj['objects'][0])
    print(len(obj['objects']))
    print(obj['embeddings'].shape)
    print(obj['embeddings'])
    emb = obj['embeddings']
    words = obj['objects']
    outfile = outfilelist[0]+'.txt'
    print('write %s' % outfile)
    with open(outfile, 'w', encoding='utf-8') as fp:
        rows = int(emb.shape[0])
        dims = int(emb.shape[1])
        fp.write('%d %d\n' % (rows, dims))
        ewords = 0
        for r in range(rows):
            li = emb[r, :]
            newli = []
            for c in range(dims):
                newli.append('%1.12f' % li[c])
            if len(words[r].split(' ')) > 1:
                fp.write('english%d %s\n' % (ewords, ' '.join(newli)))
                ewords += 1
            else:
                fp.write('%s %s\n' % (words[r], ' '.join(newli)))


if __name__ == '__main__':
    main(sys.argv[1], os.path.splitext(sys.argv[1]))
