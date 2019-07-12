#!/usr/bin/env python3

import os
import sys
import subprocess

pcall_base = ['python3', 'embed.py']
# pcall_base_options_dict = {'-epochs': '500',
#                            '-negs': '50',
#                            '-dampening': '0.75',
#                            '-ndproc': '4',
#                            '-eval_each': '1',
#                            '-fresh': '',
#                            '-sparse': '',
#                            '-gpu': '0',
#                            '-burnin_multiplier': '0.01',
#                            '-neg_multiplier': '0.1',
#                            '-lr_type': 'constant',
#                            '-train_threads': '1',
#                            '-dampening':  '1.0',
#                            '-batchsize': '10'}

# pcall_base_options_dict = {'-epochs': '1000',
#                            '-negs': '50',
#                            '-ndproc': '4',
#                            '-eval_each': '100',
#                            '-fresh': '',
#                            '-sparse': '',
#                            '-gpu': '0',
#                            '-burnin_multiplier': '0.01',
#                            '-neg_multiplier': '0.1',
#                            '-lr_type': 'constant',
#                            '-train_threads': '1',
#                            '-dampening':  '1.0',
#                            '-batchsize': '10'}
pcall_base_options_dict = {'-epochs': '50',
                           '-negs': '50',
                           '-ndproc': '4',
                           '-eval_each': '10',
                           '-fresh': '',
                           '-sparse': '',
                           '-gpu': '0',
                           '-burnin_multiplier': '0.01',
                           '-neg_multiplier': '0.1',
                           '-lr_type': 'scheduled',
                           '-train_threads': '1',
                           '-dampening':  '1.0',
                           '-batchsize': '30',
                           '-lr_step_size': '15',
                           '-lr_gamma': '0.25'}
# pcall_option_poincare = {'-lr': '0.3'}
# pcall_option_poincare = {'-lr': '1.0'}
pcall_option_poincare = {'-lr': '3.0'}
pcall_option_lorentz = {'-lr': '0.5', '-no-maxnorm': ''}
# pcall_option_lorentz = {'-lr': '0.5'}


def dict2array(dict_of_command={}):
    ret = []
    for k, v in dict_of_command.items():
        ret.append(k)
        if v != '':
            ret.append(v)

    return ret


def make_command_poincare(manifold='poincare', dim=10, dset='', checkpoint=''):
    cmd_array = pcall_base[:]
    cmd_array += dict2array(dict_of_command=pcall_base_options_dict)
    cmd_array += ['-manifold', manifold, '-dim', dim,
                  '-checkpoint', checkpoint, '-dset', dset]
    if manifold == 'lorentz':
        cmd_array += dict2array(dict_of_command=pcall_option_lorentz)
    else:
        cmd_array += dict2array(dict_of_command=pcall_option_poincare)

    return cmd_array


def main():
    # manifolds = ['euclidean', 'poincare', 'lorentz']
    # dimensions = [5, 10, 20, 50, 100, 200]
    manifolds = ['poincare']
    # manifolds = ['lorentz']
    dimensions = [3]
    # dataset = './wordnet/noun_closure_synsets_jp.csv'
    dataset = './wordnet/noun_closure_jp.csv'
    for mani in manifolds:
        for dim in dimensions:
            outfile = './pojwords3/emb_%s_%d.bin' % (mani, dim)
            cmd_python3 = make_command_poincare(
                manifold=mani, dim=str(dim), dset=dataset, checkpoint=outfile)
            print(cmd_python3)
            sts = subprocess.call(' '.join(cmd_python3), shell=True)


if __name__ == '__main__':
    main()
