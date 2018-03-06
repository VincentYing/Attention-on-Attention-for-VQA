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
from Models import Model, BaseLine

Model_Variable = BaseLine


"""
Name: plot_results()

Plots train & validation: loss, accuracy
"""
def plot_resulst(train_accuracy, train_loss, val_accuracy_per_epoch, val_loss_per_epoch, batch_size, total_epochs):

    train_Y_accuracies = [ sum(ep)/len(ep) for ep in train_accuracy]
    train_Y_loss = [ sum(ep)/len(ep) for ep in train_loss]

    train_Y_accuracies_batch = list(chain(*train_accuracy))
    train_Y_loss_batch = list(chain(*train_loss))

    num_batch = len(train_Y_loss_batch)
    num_epoch = len(val_loss_per_epoch)

    X_Batch = np.arange(1/batch_size, (num_batch+1)/batch_size, 1/batch_size)
    X_Epoch = np.arange(1, (num_epoch+1))

    plt.plot(X_Batch, train_Y_accuracies_batch, color='b', label='train accuracy')
    plt.plot(X_Epoch, train_Y_accuracies, color='b', label='train accuracy')
    plt.plot(X_Epoch, val_accuracy_per_epoch, color='g', label='val accuracy')
    plt.xlim(0, total_epochs+1)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Train & Val Accuracy')
    plt.show()

    plt.plot(X_Batch, train_Y_loss_batch, color='b', label='train loss')
    plt.plot(X_Epoch, train_Y_loss, color='b', label='train loss')
    plt.plot(X_Epoch, val_loss_per_epoch, color='g', label='val loss')
    plt.xlim(0, total_epochs + 1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Train & Val Loss')
    plt.show()


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



    #Trying stuff out


    train_loss_split = [[1,2],[1,2],[1,2],[1,2],[2,3]]  # will be a list of lists => [ epoch1[], epoch2[], ...]
    train_accuracy_split = [[1,2],[2,2],[2,2],[1,2],[1,2]]  # will be a list of lists => [ epoch1[], epoch2[], ...]
    val_loss_per_epoch = [1,2,3,4,4]  # will be a list => [ epoch1, epoch2, ...]
    val_accuracy_per_epoch = [1,2,3,3,4]

    plot_resulst(train_accuracy_split, train_loss_split, val_accuracy_per_epoch, val_loss_per_epoch, 2, 5)

    tbs = {
        'epoch': 00,
    }
    torch.save(tbs, ('%s/model-' + str(0) + '.pth.tar'%str(Model_Variable)) )



    return






    # Load Data
    print ('Loading data for training ')
    loader = Data_loader(batch_size=args.bsize, emb_dim=args.emb, multilabel=args.multilabel,
                         train=True, val=False, test=False)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d\n\toutput dim: %d' % (loader.q_words, args.emb, loader.K, loader.feat_dim,
                args.hid, loader.n_answers))

    print('Loading data for validation ')
    # batch_size=0 is a special case to process all data
    validation_loader = Data_loader(batch_size=0, emb_dim=args.emb, multilabel=args.multilabel,
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
    print('Start training.')
    train_loss_split = []       # will be a list of lists => [ epoch1[], epoch2[], ...]
    train_accuracy_split = []   # will be a list of lists => [ epoch1[], epoch2[], ...]
    val_loss_per_epoch = []  # will be a list => [ epoch1, epoch2, ...]
    val_accuracy_per_epoch = [] # will be a list => [ epoch1, epoch2, ...]

    for ep in range(args.ep):
        train_loss_split.append([])
        train_accuracy_split.append([])

        for step in range(loader.n_batches):
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
        for step in range(validation_loader.n_batches):
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

            val_loss_per_epoch.append(loss.data[0])
            val_accuracy_per_epoch.append(accuracy)


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
                'Epoch %02d done, average train loss: %.3f, average train accuracy: %.2f%%, average val loss: %.3f, average val accuracy: %.2f%%' %
               (ep+1, sum(train_loss_split[ep])/len(train_loss_split[ep]), sum(train_accuracy_split[ep])/len(train_accuracy_split[ep]),
                val_loss_per_epoch[ep], val_accuracy_per_epoch[ep])
        )




