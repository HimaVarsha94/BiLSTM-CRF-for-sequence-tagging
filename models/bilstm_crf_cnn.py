import torch
import torch.autograd as autograd
import torch.nn as nn

from models.lstm_cnn import BILSTM_CNN
from models.crf import CRF

class BiLSTM_CRF_CNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, char_size, pretrained_weight_embeddings, USE_CRF, BIDIRECTIONAL=False, USE_BIGRAM=False, bigram_size=0, CNN=False, use_gpu=0):
        super(BiLSTM_CRF_CNN, self).__init__()
        # include start and end tags
        self.gpu = use_gpu
        self.bidirectional = BIDIRECTIONAL
        self.lstm_cnn = BILSTM_CNN(embedding_dim, hidden_dim, vocab_size, tagset_size, char_size, pretrained_weight_embeddings, USE_CRF=False,
                     BIDIRECTIONAL=self.bidirectional, USE_BIGRAM=False, bigram_size=0, CNN=False, use_gpu=0)
        self.crf = CRF(tagset_size, self.gpu)

    def neg_ll_loss(self, sentence, gold_labels, chars):
        feats = self.lstm_cnn.forward(sentence, chars)
        return self.crf.neg_ll_loss(sentence, gold_labels, feats)

    def forward(self, sentence, feats):
        feats = self.lstm.forward(sentence)
        score, tag_seq = self.crf.forward(sentence, feats)
        return score, tag_seq
