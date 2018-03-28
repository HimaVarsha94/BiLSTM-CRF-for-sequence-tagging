import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(1)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_weight_embeddings, USE_CRF=False):
        super(LSTMTagger, self).__init__()
        self.tagset_size = tagset_size
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weight.requires_grad = False
        print("Entered!!!!")
        # if pretrained_weight_embeddings != None:
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight_embeddings))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

        # CRF transition matrix
        #TODO: try init randomly as well, see which works best
        self.trans_mat = np.zeros((tagset_size,tagset_size))

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores

    def forward_backward(self, state_probs, sent_len):
        c = np.zeros(sent_len)
        alpha = np.zeros((sent_len,self.tagset_size))
        beta = np.zeros((sent_len,self.tagset_size))
        epsilon = np.zeros((sent_len,self.tagset_size,self.tagset_size))

        # alphas
        for t in range(sent_len):
            for i in range(self.tagset_size):
                if t == 0:
                    alpha[t][i] = 0.0
                else:
                    for j in range(self.tagset_size):
                        alpha[t][i] += alpha[t-1][j] * self.trans_mat[i][j]
                    alpha[t][i] *= state_probs[t][i]

            alpha_sum = 0.0
            for i in range(self.tagset_size):
                alpha_sum += alpha[t][i]
            c[t] = 1.0/alpha_sum
            for i in range(self.tagset_size):
                alpha[t][i] /= alpha_sum

        # betas
        for t in range(sent_len-1,-1,-1):
            for i in range(self.tagset_size):
                if t == sent_len-1:
                    beta[0][i] = 1.0
                else:
                    for j in range(self.tagset_size):
                        beta[t][i] += beta[t+1][j] * self.trans_mat[j][i]
                    ## lolol same prob (t+1 or t)?
                    beta[t][i] *= state_probs[t+1][i]

            for i in range(self.tagset_size):
                beta[t][i] *= c[t]

        # epsilons
        for t in range(sent_len):
            for i in range(self.tagset_size):
                for j in range(self.tagset_size):
                    if t == 0:
                        epsilon[t][i][j] = 0.0
                    else:
                        # some places say that state_probs and beta should be j+1
                        epsilon[t][i][j] = alpha[t-1][j] * self.trans_mat[i][j] * state_probs[t][j] * beta[t][i]

        # update transition matrix
        for i in range(self.tagset_size):
            for t in range(sent_len):
                for j in range(self.tagset_size):
                    self.trans_mat[i][j] += epsilon[t][i][j]

    # is this right?
    def viterbi(self):
        pass
