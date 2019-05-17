#!/usr/bin/env python3

import pickle
import logging
import argparse
import numpy as np
import torch
import os
import sys
import timeit
import vecto.embeddings
import vecto.benchmarks.analogy.analogy


def read_txt_file(filename=''):
    with open(filename, 'r', encoding='utf-8') as pf:
        rows, columns = list(map(int, pf.readline().split(' ')))
        return_dict = {}
        for r in range(rows):
            list_of_line = pf.readline().split(' ')
            headword, vector = list_of_line[0], ' '.join(list_of_line[1:])
            return_dict[headword] = vector

    return return_dict


def save_txt_file_with_selected_words(filename='', selected=[], data_dict={}):
    rows = len(selected)
    columns = len(data_dict[selected[0]].split(' '))
    with open(filename, 'w', encoding='utf-8') as pf:
        pf.write('%d %d\n' % (rows, columns))
        for r in range(rows):
            word = selected[r]
            str_of_line = ' '.join([word, data_dict[word]])
            pf.write('%s' % str_of_line)

    print('%s: %d records are saved.' % (filename, rows))

def main(poincare='', w2v=''):
    pref_poincare, ext_poincare = os.path.splitext(poincare)
    pref_w2v, ext_w2v = os.path.splitext(w2v)

    out_poincare = pref_poincare + '_cl' + ext_poincare
    out_w2v = pref_w2v + '_cl' + ext_w2v

    poincare_dict = read_txt_file(filename=poincare)
    w2v_dict = read_txt_file(filename=w2v)

    p_keys = set(poincare_dict.keys())
    w_keys = set(w2v_dict.keys())

    out_keys = list(p_keys & w_keys)
    save_txt_file_with_selected_words(
        filename=out_poincare, selected=out_keys, data_dict=poincare_dict)
    save_txt_file_with_selected_words(
        filename=out_w2v, selected=out_keys, data_dict=w2v_dict)


if __name__ == '__main__':
    main(poincare=sys.argv[1], w2v=sys.argv[2])
