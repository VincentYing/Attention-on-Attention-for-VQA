from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import time
import json
import argparse


from train import train
from test import test
from val import val


"""
Name: parse_args 

Parses user's arguements 
"""
def parse_args():
    parser = argparse.ArgumentParser(description='Winner of VQA 2.0 in CVPR\'17 Workshop')
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--eval', action='store_true', help='set this to evaluate on validation set')
    parser.add_argument('--test', action='store_true', help='set this to evaluate on TEST set')
    parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--ep', metavar='', type=int, default=50, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=512, help='batch size.')
    parser.add_argument('--hid', metavar='', type=int, default=512, help='hidden dimension.')
    parser.add_argument('--emb', metavar='', type=int, default=300, help='embedding dimension. (50, 100, 200, *300)')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained model path.')
    parser.add_argument('--multilabel', metavar='', type=bool, default=True, help='set this to use multilabel.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))

    return args, parser


"""
Name: main 

Takes in user arguments such as train, eval, hyperparams
Calls train, test or eval
"""
if __name__ == '__main__':
    args, parser = parse_args()

    if args.train:
        train(args)
    elif args.eval:
        val(args)
    elif args.test:
        test(args)
    if not args.train and not args.eval and not args.test:
        parser.print_help()

