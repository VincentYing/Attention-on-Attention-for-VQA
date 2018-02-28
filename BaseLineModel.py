from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLineModel(nn.Module): #Change the name for each model

    """
    This is where you make all the variable of the model
    """
    def __init__(self, vocab_size, emb_dim, K, feat_dim, hid_dim, out_dim, pretrained_wemb):

        super(BaseLineModel, self).__init__()
        """
        Args:
            vocab_size: vocabularies to embed (question vocab size)
            emb_dim: GloVe pre-trained dimension -> 300
            K: image bottom-up attention locations, aka number of image features -> 36
            feat_dim: image feature dimension -> 2048
            hid_dim: hidden dimension -> 512
            out_dim: multi-label regression output -> (answer vocab size)
            pretrained_wemb: pretrained word embeddings (vocab_size, emb_dim)
        """


        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.K = K
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim


        """
        #question encoding seperate embedding for each word
        self.wembed = nn.Embedding(num_embeddings= self.vocab_size, embedding_dim= self.emb_dim)
        # initialize word embedding layer weights
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))
        """


        # question encoding seperate embedding for each word
        self.wembed_BOG = nn.EmbeddingBag(num_embeddings=self.vocab_size, embedding_dim= self.emb_dim, mode= 'mean')
        # initialize word embedding layer weights
        self.wembed_BOG.weight.data.copy_(torch.from_numpy(pretrained_wemb))


        #First 1 layer network, with outputlayer [for attention]
        self.NN1_W1 = nn.Linear(in_features=self.emb_dim + self.feat_dim, out_features=self.hid_dim, bias=True)
        self.NN1_W2 = nn.Linear(in_features=self.hid_dim, out_features=1, bias=True)

        #Second 2 layer network, with outputlayer [for answer]
        self.NN2_W1 = nn.Linear(in_features=self.emb_dim + self.feat_dim, out_features=self.hid_dim, bias=True)
        self.NN2_W2 = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim, bias=True)
        self.NN2_W3 = nn.Linear(in_features=self.hid_dim, out_features=self.out_dim, bias=True)




    def forward(self, question, image):
        """
        question -> shape (batch, seqlen)
        image -> shape (batch, K, feat_dim)
        """

        # Get question encoding (Words averaged) Bag of Words
        q_enc = self.wembed_BOG(question)       # (batch, emb_dim)

        # image encoding
        image = F.normalize(image, -1)          # (batch, K, feat_dim)

        # image attention
        q_enc_reshape = q_enc.repeat(1, self.K).view(-1, self.K, self.emb_dim)  # (batch, K, emb_dim)
        concat_1 = torch.cat((image, q_enc_reshape), -1)        # (batch, K, feat_dim + emb_dim)
        concat_1 = F.relu(self.NN1_W1(concat_1))                # (batch, K, hid_dim)
        attention = self.NN1_W2(concat_1)                       # (batch, K, 1)
        attention = F.softmax(attention.squeeze(), dim=1)       # (batch, K)

        # get weighted image vector
        context_vec = torch.bmm(attention.unsqueeze(1), image).squeeze()  # (batch, feat_dim)

        # Output
        concat_2 = torch.cat((context_vec, q_enc), -1)      # (batch, feat_dim + emb_dim)
        concat_2 = F.relu(self.NN2_W1(concat_2))            # (batch, hid_dim)
        concat_2 = F.relu(self.NN2_W2(concat_2))            # (batch, hid_dim)
        output = self.NN2_W3(concat_2)                      # (batch, out_dim)

        return output               # (batch, out_dim)

