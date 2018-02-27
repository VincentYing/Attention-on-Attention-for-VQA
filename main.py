from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from itertools import chain
import matplotlib.pyplot as plt


from loader import Data_loader

"""
Import Models  
"""
from model import Model
from BaseLineModel import BaseLineModel
from Model_1 import Model_1

Model_Variable = BaseLineModel



"""
Name: test

This function tests the model on the TEST Set. 
Only do this at the end. 
"""
def test(args):
    # Some preparation
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, don\'t do this.')

    print ('Loading data')
    loader = Data_loader(batch_size=args.bsize, emb_dim=args.emb, multilabel=args.multilabel,
                         train=False, val=False, test=True)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d\n\toutput dim: %d' % (loader.q_words, args.emb, loader.K, loader.feat_dim,
                args.hid, loader.n_answers))

    # chose model & build its graph, Model chosen above in global variable
    model = Model_Variable(vocab_size=loader.q_words,
                  emb_dim=args.emb,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=loader.n_answers,
                  pretrained_wemb=loader.pretrained_wemb)

    model = model.cuda()

    if args.modelpath and os.path.isfile(args.modelpath):
        print ('Resuming from checkpoint %s' % (args.modelpath))
        ckpt = torch.load(args.modelpath)
        model.load_state_dict(ckpt['state_dict'])
    else:
        raise SystemExit('Need to provide model path.')

    result = []
    for step in xrange(loader.n_batches):
        # Batch preparation
        q_batch, a_batch, i_batch = loader.next_batch()
        q_batch = Variable(torch.from_numpy(q_batch))
        i_batch = Variable(torch.from_numpy(i_batch))
        q_batch, i_batch = q_batch.cuda(), i_batch.cuda()

        # Do one model forward and optimize
        output = model(q_batch, i_batch)
        _, ix = output.data.max(1)
        for i, qid in enumerate(a_batch):
            result.append({
                'question_id': qid,
                'answer': loader.a_itow[ix[i]]
            })

    json.dump(result, open('result.json', 'w'))
    print ('Validation done')








"""
Name: val

This function tests the model on the validation set
This function does this directly.
However the model is also tested 
"""
def val(args):
    # Some preparation
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, don\'t do this.')

    print ('Loading data')
    loader = Data_loader(batch_size=args.bsize, emb_dim=args.emb, multilabel=args.multilabel,
                         train=False, val=True, test=False)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d\n\toutput dim: %d' % (loader.q_words, args.emb, loader.K, loader.feat_dim,
                args.hid, loader.n_answers))

    # chose model & build its graph, Model chosen above in global variable
    model = Model_Variable(vocab_size=loader.q_words,
                  emb_dim=args.emb,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=loader.n_answers,
                  pretrained_wemb=loader.pretrained_wemb)

    model = model.cuda()

    if args.modelpath and os.path.isfile(args.modelpath):
        print ('Resuming from checkpoint %s' % (args.modelpath))
        ckpt = torch.load(args.modelpath)
        model.load_state_dict(ckpt['state_dict'])
    else:
        raise SystemExit('Need to provide model path.')

    result = []
    for step in xrange(loader.n_batches):
        # Batch preparation
        q_batch, a_batch, i_batch = loader.next_batch()
        q_batch = Variable(torch.from_numpy(q_batch))
        i_batch = Variable(torch.from_numpy(i_batch))
        q_batch, i_batch = q_batch.cuda(), i_batch.cuda()

        # Do one model forward
        output = model(q_batch, i_batch)

        # Do evaluation
        _, ix = output.data.max(1)
        for i, qid in enumerate(a_batch):
            result.append({
                'question_id': qid,
                'answer': loader.a_itow[ix[i]]
            })

    json.dump(result, open('result.json', 'w'))
    print ('Validation done')




"""
Name: train

Adam optimizer currently 
"""
def train(args):
    # Some preparation
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, don\'t do this.')

    # Load Data
    print ('Loading data for training ')
    loader = Data_loader(batch_size=args.bsize, emb_dim=args.emb, multilabel=args.multilabel,
                         train=True, val=False, test=False)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d\n\toutput dim: %d' % (loader.q_words, args.emb, loader.K, loader.feat_dim,
                args.hid, loader.n_answers))
    print('Loading data for validation ')
    # batch_size=0 is a special case to process all data
    validation_loader = Data_loader(batch_size=args.bsize, emb_dim=args.emb, multilabel=args.multilabel,
                         train=False, val=True, test=False)


    # Chose model & build its graph, Model chosen above in global variable
    print('Initializing model')
    model = Model_Variable(vocab_size=loader.q_words,
                  emb_dim=args.emb,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=loader.n_answers,
                  pretrained_wemb=loader.pretrained_wemb)

    # Classification and Loss Function
    if args.multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()


    # Move it to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    """
    optimizer = torch.optim.Adamax(model.parameters())
    """

    # Continue training from saved model
    if args.modelpath and os.path.isfile(args.modelpath):
        print ('Resuming from checkpoint %s' % (args.modelpath))
        ckpt = torch.load(args.modelpath)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Training script 
    print ('Start training.')
    train_loss_split = [] # will be a list of lists => [ epoch1[], epoch2[], ...]
    val_loss_per_epoch = [] # will be a list of lists => [ epoch1[], epoch2[], ...]
    train_accuracy_split = []  # will be a list of lists => [ epoch1[], epoch2[], ...]
    val_accuracy_per_epoch = []

    """
    from itertools import chain
    newlist = list(chain(*newlist))
    """

    for ep in xrange(args.ep):
        train_loss_split.append([])

        for step in xrange(loader.n_batches):
            # Batch preparation
            q_batch, a_batch, i_batch = loader.next_batch()
            q_batch = Variable(torch.from_numpy(q_batch))
            a_batch = Variable(torch.from_numpy(a_batch))
            i_batch = Variable(torch.from_numpy(i_batch))
            q_batch, a_batch, i_batch = q_batch.cuda(), a_batch.cuda(), i_batch.cuda()

            # Do model forward
            output = model(q_batch, i_batch)
            loss = criterion(output, a_batch)

            # compute gradient and do optim step
            loss.backward()
            """
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            """
            optimizer.step()
            optimizer.zero_grad()


            # Some stats
            _, oix = output.data.max(1)
            if args.multilabel:
                _, aix = a_batch.data.max(1)
            else:
                aix = a_batch.data
            correct = torch.eq(oix, aix).sum()
            accuracy = correct*100/args.bsize

            train_loss_split[ep].append(loss.data[0])
            train_accuracy_split[ep].append(accuracy)

            if step % 40 == 0:
                print ('Epoch %02d(%03d/%03d), loss: %.3f, correct: %3d / %d, accuracy: (%.2f%%)' %
                        (ep+1, step, loader.n_batches, loss.data[0], correct, args.bsize, correct * 100 / args.bsize))


        """
        Run Validation Here 
        """
        for step in xrange(validation_loader.n_batches):
            # All Validation set preparation
            q, a, i = validation_loader.next_batch()
            q = Variable(torch.from_numpy(q))
            a = Variable(torch.from_numpy(a))
            i = Variable(torch.from_numpy(i))
            q, a, i = q.cuda(), a.cuda(), i.cuda()

            # Do model forward
            output = model(q, i)
            loss = criterion(output, a)


            # Some stats
            _, oix = output.data.max(1)
            if args.multilabel:
                _, aix = a.data.max(1)
            else:
                aix = a.data
            correct = torch.eq(oix, aix).sum()
            accuracy = correct*100/validation_loader.bsize

            val_loss_per_epoch[ep].append(loss.data[0])
            val_accuracy_per_epoch[ep].append(accuracy)


        # Save model after every epoch
        tbs = {
            'epoch': ep + 1,
            'train_loss': train_loss_split,
            'train_accuracy': train_accuracy_split,
            'val_loss': val_loss_per_epoch,
            'val_accuracy':val_accuracy_per_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(tbs, 'save/model-' + str(ep+1) + '.pth.tar')

        print (
                'Epoch %02d done, average train loss: %.3f, average train accuracy: %.2f%%, average train loss: %.3f, average train accuracy: %.2f%%' %
               (ep+1, sum(train_loss_split[ep])/len(train_loss_split[ep]), sum(train_accuracy_split[ep])/len(train_accuracy_split[ep]),
                sum(val_loss_per_epoch[ep])/len(val_loss_per_epoch[ep]), sum(val_accuracy_per_epoch[ep])/len(val_accuracy_per_epoch[ep]))
        )


        """
        Plot Results Here 
        """
        X_batch = np.arange(1,(args.ep+1),1/loader.n_batches)
        X_batch_2 = np.arange(1, (args.ep + 1),1/validation_loader.n_batches)
        #X_epoch = range(1,(args.ep+1))

        train_Y_batch_accuracies = list(chain(*train_accuracy_split))
        train_Y_batch_loss = list(chain(*train_loss_split))

        val_Y_accuracies = list(chain(*val_accuracy_per_epoch))
        val_Y_loss = list(chain(*val_loss_per_epoch))

        plt.plot(X_batch,train_Y_batch_accuracies, color='b', label ='train accuracy')
        plt.plot(X_batch_2, val_Y_accuracies, color='g', label='val accuracy')
        plt.show()

        plt.plot(X_batch, train_Y_batch_loss, color='b', label='train loss')
        plt.plot(X_batch_2, val_Y_loss, color='g', label='val loss')
        plt.show()





"""
Name: Main 

Takes in user arguments such train, eval, hyperparams
Calls Train or Test above 
"""
if __name__ == '__main__':
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
    if args.train:
        train(args)
    if args.eval:
        val(args)
    if args.test:
        test(args)
    if not args.train and not args.eval:
        parser.print_help()

