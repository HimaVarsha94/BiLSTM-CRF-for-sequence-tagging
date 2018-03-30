import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.crf import CRF

torch.manual_seed(1)


class BILSTM_CNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, char_size, pretrained_weight_embeddings, tag_to_ix, USE_CRF=False,
                 BIDIRECTIONAL=False, USE_BIGRAM=False, bigram_size=0, CNN=False, use_gpu=0):
        super(BILSTM_CNN, self).__init__()
        self.char_dim = 25
        self.char_lstm_dim = 25
        self.CNN = CNN
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weight.requires_grad = False
        # if pretrained_weight_embeddings != None:
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight_embeddings))



        if self.CNN:
            print("Entered!!!!")
            self.char_embeds = nn.Embedding(char_size, self.char_dim)
            self.init_embedding(self.char_embeds.weight)
            self.char_cnn = nn.Conv2d(in_channels=1, out_channels=self.char_lstm_dim, kernel_size=(3, self.char_dim), padding=(2,0))

        if BIDIRECTIONAL:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=BIDIRECTIONAL)
        else:
            self.lstm = nn.LSTM(embedding_dim + self.char_lstm_dim, hidden_dim)

        self.dropout = nn.Dropout(0.5)
        self.bidirectional = BIDIRECTIONAL
        self.append_bigram = USE_BIGRAM
        self.hidden = self.init_hidden()
        if self.append_bigram:
            self.hidden2tag = nn.Linear(hidden_dim + bigram_size, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.crf = CRF(tagset_size, tag_to_ix, self.use_gpu)

    def init_hidden(self):
        if self.use_gpu:
            if not self.bidirectional:
                return (autograd.Variable(torch.randn(1, 1, self.hidden_dim).cuda()),
                        autograd.Variable(torch.randn(1, 1, self.hidden_dim)).cuda())
            else:
                return (autograd.Variable(torch.randn(2, 1, self.hidden_dim ).cuda()),
                        autograd.Variable(torch.randn(2, 1, self.hidden_dim )).cuda())
        else:
            if not self.bidirectional:
                return (autograd.Variable(torch.randn(1, 1, self.hidden_dim)),
                        autograd.Variable(torch.randn(1, 1, self.hidden_dim)))
            else:
                return (autograd.Variable(torch.cuda.randn(2, 1, self.hidden_dim )),
                        autograd.Variable(torch.cuda.randn(2, 1, self.hidden_dim )))

    def forward_lstm(self, sentence, chars, bigram_one_hot=None):
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence)  # shape seq_length * emb_size
        if self.CNN == True:
            chars_embeds = self.char_embeds(chars).unsqueeze(1)
            cnn_output = self.char_cnn(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(cnn_output,
                                                 kernel_size=(cnn_output.size(2), 1)).view(cnn_output.size(0), self.char_lstm_dim)
            if self.use_gpu:
                embeds = torch.cat((embeds, chars_embeds), 1).cuda()
            else:
                embeds = torch.cat((embeds, chars_embeds), 1)
        # print(embeds.view(len(sentence), 1, -1).shape)
        lstm_out, self.hidden = self.lstm(embeds.unsqueeze(1), self.hidden)
        # print("original shape before MLP "+str(lstm_out.view(len(sentence), -1).shape))
        # print("shape of onehot bigram "+str(bigram_one_hot))
        if self.append_bigram:
            # print("concatednated vector"+str(lstm_out.view(len(sentence), -1)))
            tag_space=self.hidden2tag(torch.cat([lstm_out.view(len(sentence), -1),bigram_one_hot],dim=1))
        else:
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space


    def neg_ll_loss(self, sentence, gold_labels, chars):
        feats = self.forward(sentence, chars)
        return self.crf.neg_ll_loss(sentence, gold_labels, feats)

    def forward(self, sentence, chars):
        feats = self.forward_lstm(sentence, chars)
        score, tag_seq = self.crf.forward(sentence, feats)

        return score, tag_seq

    def init_embedding(self, input_embedding):
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform(input_embedding, -bias, bias)
