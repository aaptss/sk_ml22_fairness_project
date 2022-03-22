#!/usr/bin/env python3

from argparse import ArgumentParser
from oct2py import Oct2Py


datasets = ['adult',
            'bank',
            'census',
            'compass']

parser = ArgumentParser('datasets')
parser.add_argument('-d',
                    '--dataset',
                    type=str,
                    choices=['adult', 'bank', 'census', 'compass'],
                    action='store',
                    help='Available datasets: \'adult\', \'bank\', \'census\', \'compass\'')
args = parser.parse_args()
dataset = args.dataset


oc = Oct2Py()
oc.eval('packs')
data = oc.eval(f'choose_dataset(\'{dataset}\')')
accuracy, bal_acc, eq_odds, TPR_prot, TPR_non_prot, TNR_prot, TNR_non_prot = oc.eval(f'main({data})', nout=7)

print('Accuracy: ', accuracy)
print('Balanced accuracy: ', bal_acc)
print('Equalized Odds: ', eq_odds)
print('TPR prot.: ', TPR_prot)
print('TPR non-prot: ', TPR_non_prot)
print('TNR prot.: ', TNR_prot)
print('TNR non-prot: ', TNR_non_prot)
