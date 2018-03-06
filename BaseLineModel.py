from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLineModel(nn.Module): #Change the name for each model

    def __init__(self, vocab_size, emb_dim, K, feat_dim, hid_dim, out_dim, pretrained_wemb):
        super(BaseLineModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.K = K                  # num image locations
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim      # out_put vocab

        #question encoding seperate embedding for each word
        self.wembed = nn.Embedding(num_embeddings= self.vocab_size, embedding_dim= self.emb_dim)
        # initialize word embedding layer weights
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))

        self.gru = nn.GRU(emb_dim, hid_dim)

        # First 1 layer network, with outputlayer [for attention]
        self.NN1_W1 = nn.Linear(in_features=self.emb_dim + self.hid_dim, out_features=self.hid_dim, bias=True)
        self.NN1_W2 = nn.Linear(in_features=self.hid_dim, out_features=1, bias=True)

        # Second 2 layer network, with outputlayer [for answer]
        self.NN2_W1 = nn.Linear(in_features=self.hid_dim + self.feat_dim, out_features=self.hid_dim, bias=True)
        self.NN2_W2 = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim, bias=True)
        self.NN2_W3 = nn.Linear(in_features=self.hid_dim, out_features=self.out_dim, bias=True)


    def forward(self, question, image):
        """
        question -> shape (batch, seqlen)
        image -> shape (batch, K, feat_dim)
        """
        emb = self.wembed(question)                 # (batch, seqlen, emb_dim)
        enc, hid = self.gru(emb.permute(1, 0, 2))   # (seqlen, batch, hid_dim)
        q_enc = enc[-1]  # (batch, hid_dim)

        image = F.normalize(image, -1)          # (batch, K, feat_dim)

        # image attention
        q_enc_reshape = q_enc.repeat(1, self.K).view(-1, self.K, self.emb_dim)  # (batch, K, hid_dim)
        print(q_enc_reshape.size(),"(batch, K, hid_dim)")

        concat_1 = torch.cat((image, q_enc_reshape), -1)        # (batch, K, feat_dim + hid_dim)
        print(concat_1.size(), "(batch, K, feat_dim + hid_dim)")

        concat_1 = F.relu(self.NN1_W1(concat_1))                # (batch, K, hid_dim)
        print(concat_1.size(), "(batch, K, hid_dim)")

        attention = self.NN1_W2(concat_1)                       # (batch, K, 1)
        print(attention.size(), "(batch, K, 1)")
        
        attention = F.softmax(attention, dim=1)                 # (batch, K, 1)
        print(attention.size(), "(batch, K, 1)")

        context_vec = (attention * image).sum(1)           # (batch, feat_dim): (batch, K, 1) * (batch, K, feat_dim)
        print(context_vec.size(),  "(batch, feat_dim)")

        # Output
        concat_2 = torch.cat((context_vec, q_enc), -1)      # (batch, feat_dim + hid_dim)
        concat_2 = F.relu(self.NN2_W1(concat_2))            # (batch, hid_dim)
        concat_2 = F.relu(self.NN2_W2(concat_2))            # (batch, hid_dim)
        output = self.NN2_W3(concat_2)                      # (batch, out_dim)

        return output               # (batch, out_dim)
