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

        #question encoding
        self.wembed = nn.Embedding(self.vocab_size, self.emb_dim)
        # initialize word embedding layer weights
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))


        #First 2 layer network, with outputlayer [for attention]
        self.NN1_W1 = nn.Linear(in_features=self.emb_dim + self.feat_dim, out_features=self.hid_dim, bias=True)
        self.NN1_W2 = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim, bias=True)
        self.NN1_W3 = nn.Linear(in_features=self.hid_dim, out_features=1, bias=True)

        #Second 2 layer network, with outputlayer [for answer]
        self.NN2_W1 = nn.Linear(in_features=self.emb_dim + self.feat_dim, out_features=self.hid_dim, bias=True)
        self.NN2_W2 = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim, bias=True)
        self.NN2_W3 = nn.Linear(in_features=self.hid_dim, out_features=self.out_dim, bias=True)






        """
        OLD CODE 
        
        # gated tanh activation
        self.gt_W_img_att = nn.Linear(feat_dim + hid_dim, hid_dim)
        self.gt_W_prime_img_att = nn.Linear(feat_dim + hid_dim, hid_dim)
        self.gt_W_question = nn.Linear(hid_dim, hid_dim)
        self.gt_W_prime_question = nn.Linear(hid_dim, hid_dim)
        self.gt_W_img = nn.Linear(feat_dim, hid_dim)
        self.gt_W_prime_img = nn.Linear(feat_dim, hid_dim)
        self.gt_W_clf = nn.Linear(hid_dim, hid_dim)
        self.gt_W_prime_clf = nn.Linear(hid_dim, hid_dim)
        # question encoding
        self.wembed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        # image attention
        self.att_wa = nn.Linear(hid_dim, 1)
        # output classifier
        self.clf_w = nn.Linear(hid_dim, out_dim)
        # initialize word embedding layer weight
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))
        """


    def forward(self, question, image):
        """
        question -> shape (batch, seqlen)
        image -> shape (batch, K, feat_dim)
        """
        # get question encoding, then sum together: Bag of Words
        emb = self.wembed(question)                         # (batch, seqlen, emb_dim)
        q_enc = torch.sum(input=emb, dim=1, keepdim=False)  # (batch, emb_dim)

        # image encoding
        image = F.normalize(image, -1)  # (batch, K, feat_dim)

        # image attention
        q_enc_reshape = q_enc.repeat(1, self.K).view(-1, self.K, self.emb_dim)  # (batch, K, emb_dim)
        concat_1 = torch.cat((image, q_enc_reshape), -1)        # (batch, K, feat_dim + emb_dim)
        concat_1 = F.tanh(self.NN1_W1(concat_1))                # (batch, K, hid_dim)
        concat_1 = F.tanh(self.NN1_W2(concat_1))                # (batch, K, hid_dim)
        attention = self.NN1_W3(concat_1)                       # (batch, K, 1)
        attention = F.softmax(attention.squeeze())              # (batch, K)

        # get weighted image vector
        context_vec = torch.bmm(attention.unsqueeze(1), image).squeeze()  # (batch, feat_dim)

        # Output
        concat_2 = torch.cat((context_vec, q_enc), -1)      # (batch, feat_dim + emb_dim)
        concat_2 = F.tanh(self.NN2_W1(concat_2))            # (batch, hid_dim)
        concat_2 = F.tanh(self.NN2_W2(concat_2))            # (batch, hid_dim)
        output = self.NN2_W3(concat_2)                      # (batch, out_dim)

        """
        OlD CODE 
        
        # question encoding
        emb = self.wembed(question)                 # (batch, seqlen, emb_dim)
        enc, hid = self.gru(emb.permute(1, 0, 2))   # (seqlen, batch, hid_dim)
        qenc = enc[-1]                              # (batch, hid_dim)
        # image encoding
        image = F.normalize(image, -1)  # (batch, K, feat_dim)
        # image attention
        qenc_reshape = qenc.repeat(1, self.K).view(-1, self.K, self.hid_dim)    # (batch, K, hid_dim)
        concated = torch.cat((image, qenc_reshape), -1)                         # (batch, K, feat_dim + hid_dim)
        concated = self._gated_tanh(concated, self.gt_W_img_att, self.gt_W_prime_img_att)   # (batch, K, hid_dim)
        a = self.att_wa(concated)                           # (batch, K, 1)
        a = F.softmax(a.squeeze())                          # (batch, K)
        v_head = torch.bmm( a.unsqueeze(1), image ).squeeze() # (batch, feat_dim)
        # element-wise (question + image) multiplication
        q = self._gated_tanh(qenc, self.gt_W_question, self.gt_W_prime_question)
        v = self._gated_tanh(v_head, self.gt_W_img, self.gt_W_prime_img)
        h = torch.mul(q, v)         # (batch, hid_dim)
        # output classifier
        s_head = self.clf_w(self._gated_tanh(h, self.gt_W_clf, self.gt_W_prime_clf))
        """

        return output               # (batch, out_dim)

    def _gated_tanh(self, x, W, W_prime):
        """
        Implement the gated hyperbolic tangent non-linear activation
            x: input tensor of dimension m
        """
        y_tilde = F.tanh(W(x))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y
