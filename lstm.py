import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_weight_embeddings, USE_CRF=False, BIDIRECTIONAL=False):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weight.requires_grad = False
        print("Entered!!!!")
        # if pretrained_weight_embeddings != None:
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight_embeddings))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, bidirectional=BIDIRECTIONAL)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.bidirectional=BIDIRECTIONAL
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if not self.bidirectional:
            return (autograd.Variable(torch.randn(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.randn(1, 1, self.hidden_dim)))
        else:
            return (autograd.Variable(torch.randn(2, 1, self.hidden_dim//2)),
                    autograd.Variable(torch.randn(2, 1, self.hidden_dim//2)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) # shape seq_length * emb_size
        # print(embeds.view(len(sentence), 1, -1).shape)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        # print("finished forward score calculating")
        return tag_scores
