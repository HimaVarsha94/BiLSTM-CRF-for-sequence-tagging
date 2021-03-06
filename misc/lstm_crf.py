import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from models.crf import CRF

torch.manual_seed(1)


class LSTM_CRF(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, tag_to_ix, USE_CRF=False,
                 BIDIRECTIONAL=False, USE_BIGRAM=False, bigram_size=0):
        super(LSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weight.requires_grad = False
        print("Entered!!!!")
        # if pretrained_weight_embeddings != None:
        #  self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight_embeddings))
        if BIDIRECTIONAL:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=BIDIRECTIONAL)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.crf = CRF(tagset_size, tag_to_ix, 0)

        self.bidirectional = BIDIRECTIONAL
        self.append_bigram = USE_BIGRAM
        self.hidden = self.init_hidden()
        if self.append_bigram:
            self.hidden2tag = nn.Linear(hidden_dim + bigram_size, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)



    def init_hidden(self):
        if not self.bidirectional:
            return (autograd.Variable(torch.randn(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.randn(1, 1, self.hidden_dim)))
        else:
            return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                    autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def forward_lstm(self, sentence, bigram_one_hot=None):
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence)  # shape seq_length * emb_size
        # print(embeds.view(len(sentence), 1, -1).shape)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        # print("original shape before MLP "+str(lstm_out.view(len(sentence), -1).shape))
        # print("shape of onehot bigram "+str(bigram_one_hot))
        if self.append_bigram:
            # print("concatednated vector"+str(lstm_out.view(len(sentence), -1)))
            tag_space=self.hidden2tag(torch.cat([lstm_out.view(len(sentence), -1),bigram_one_hot],dim=1))
        else:
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space

    def neg_ll_loss(self, sentence, gold_labels):
        feats = self.forward_lstm(sentence)
        return self.crf.neg_ll_loss(sentence, gold_labels, feats)

    def forward(self, sentence):
        feats = self.forward_lstm(sentence)
        score, tag_seq = self.crf.forward(sentence, feats)
        return score, tag_seq
