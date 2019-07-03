#!/usr/bin/env python3

import os
import sys
import json
import csv


def mainroutine(jsonfile='po1500.json'):
    with open(jsonfile, 'r') as fp:
        js = json.load(fp)

    prefix, suffix = os.path.splitext(jsonfile)
    outfile = prefix + '.csv'

    with open(outfile, 'w') as csvf:
        csvwriter = csv.DictWriter(csvf, delimiter=',', quoting=csv.QUOTE_NONNUMERIC,
                                   fieldnames=list(js[0].keys()))
        csvwriter.writeheader()
        for r in js:
            csvwriter.writerow(r)

    print('done...(^^;)')


if __name__ == '__main__':
    mainroutine(jsonfile=sys.argv[1])
