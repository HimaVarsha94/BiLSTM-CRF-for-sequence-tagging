import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.crf import CRF

torch.manual_seed(1)

class BILSTM_CNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, char_size, pretrained_weight_embeddings, tag_to_ix, vocab_sizes,
                USE_CRF=False, BIDIRECTIONAL=False, USE_BIGRAM=False, bigram_size=0, CNN=False, use_gpu=0, duolingo_student=True):
        super(BILSTM_CNN, self).__init__()
        self.USE_CRF = USE_CRF
        self.char_dim = 25
        self.char_lstm_dim = 25
        self.CNN = CNN
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.n_cap = 4
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.duolingo_student = duolingo_student
        # self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight_embeddings))
        b = np.sqrt(3.0 / self.word_embeddings.weight.size(1))
        nn.init.uniform_(self.word_embeddings.weight, -b, b)
        #CHAR
        self.cap_embedding_dim = 25
        self.cap_embeds = nn.Embedding(self.n_cap, self.cap_embedding_dim)
        b = np.sqrt(3.0 / self.cap_embeds.weight.size(1))
        nn.init.uniform_(self.cap_embeds.weight, -b, b)

        if duolingo_student:
            #TODO change the student_id_unique
            self.student_id_unique = vocab_sizes['user']
            print('Student id vocab size',self.student_id_unique)
            self.student_id_dim = 100
            self.student_embeds = nn.Embedding(self.student_id_unique, self.student_id_dim)
            b = np.sqrt(3.0 / self.student_embeds.weight.size(1))
            nn.init.uniform_(self.student_embeds.weight, -b, b)

        ### More sequence embeddings
        ## countries:
        num_unique = vocab_sizes['country']
        self.country_dim = 100
        self.country_embeds = nn.Embedding(num_unique, self.country_dim)
        b = np.sqrt(3.0 / self.country_embeds.weight.size(1))
        nn.init.uniform_(self.country_embeds.weight, -b, b)
        ## days:
        num_unique = vocab_sizes['days']
        self.days_dim = 100
        self.days_embeds = nn.Embedding(num_unique, self.days_dim)
        b = np.sqrt(3.0 / self.days_embeds.weight.size(1))
        nn.init.uniform_(self.days_embeds.weight, -b, b)
        ## client:
        num_unique = vocab_sizes['client']
        self.client_dim = 100
        self.client_embeds = nn.Embedding(num_unique, self.client_dim)
        b = np.sqrt(3.0 / self.client_embeds.weight.size(1))
        nn.init.uniform_(self.client_embeds.weight, -b, b)
        ## session:
        num_unique = vocab_sizes['session']
        self.session_dim = 100
        self.session_embeds = nn.Embedding(num_unique, self.session_dim)
        b = np.sqrt(3.0 / self.session_embeds.weight.size(1))
        nn.init.uniform_(self.session_embeds.weight, -b, b)
        ## format:
        num_unique = vocab_sizes['format']
        self.format_dim = 100
        self.format_embeds = nn.Embedding(num_unique, self.format_dim)
        b = np.sqrt(3.0 / self.format_embeds.weight.size(1))
        nn.init.uniform_(self.format_embeds.weight, -b, b)
        ## time:
        num_unique = vocab_sizes['time']
        self.time_dim = 100
        self.time_embeds = nn.Embedding(num_unique, self.time_dim)
        b = np.sqrt(3.0 / self.time_embeds.weight.size(1))
        nn.init.uniform_(self.time_embeds.weight, -b, b)

        seq_feat_dims = self.student_id_dim + self.country_dim + self.days_dim + self.client_dim + self.session_dim + self.format_dim + self.time_dim

        ### Token embeddings
        ## pos:
        num_unique = vocab_sizes['pos']
        self.pos_dim = 100
        self.pos_embeds = nn.Embedding(num_unique, self.pos_dim)
        b = np.sqrt(3.0 / self.pos_embeds.weight.size(1))
        nn.init.uniform_(self.pos_embeds.weight, -b, b)
        ## edge_labels:
        num_unique = vocab_sizes['edge_labels']
        self.edge_labels_dim = 100
        self.edge_labels_embeds = nn.Embedding(num_unique, self.edge_labels_dim)
        b = np.sqrt(3.0 / self.edge_labels_embeds.weight.size(1))
        nn.init.uniform_(self.edge_labels_embeds.weight, -b, b)
        ## edge_heads:
        num_unique = vocab_sizes['edge_heads']
        self.edge_heads_dim = 100
        self.edge_heads_embeds = nn.Embedding(num_unique, self.edge_heads_dim)
        b = np.sqrt(3.0 / self.edge_heads_embeds.weight.size(1))
        nn.init.uniform_(self.edge_heads_embeds.weight, -b, b)

        token_feat_dims = self.pos_dim + self.edge_labels_dim + self.edge_heads_dim

        if self.CNN:
            self.char_embeds = nn.Embedding(char_size, self.char_dim)
            #as given in the paper, initialising
            b = np.sqrt(3.0 / self.char_embeds.weight.size(1))
            nn.init.uniform_(self.char_embeds.weight, -b, b)

            # self.init_embedding(self.char_embeds.weight)
            self.char_cnn = nn.Conv2d(in_channels=1, out_channels=self.char_lstm_dim, kernel_size=(3, self.char_dim), padding=(2,0))

        if BIDIRECTIONAL:
            print("Bidirectional")
            if not duolingo_student:
                self.lstm = nn.LSTM(embedding_dim + self.char_lstm_dim + self.cap_embedding_dim, hidden_dim, bidirectional=BIDIRECTIONAL)
            elif not CNN:
                self.lstm = nn.LSTM(embedding_dim + self.cap_embedding_dim + seq_feat_dims + token_feat_dims, hidden_dim, bidirectional=BIDIRECTIONAL)
            else:
                self.lstm = nn.LSTM(embedding_dim + self.char_lstm_dim + self.cap_embedding_dim + seq_feat_dims + token_feat_dims, hidden_dim, bidirectional=BIDIRECTIONAL)
        else:
            self.lstm = nn.LSTM(embedding_dim + self.char_lstm_dim, hidden_dim)

        self.drop_probout = nn.Dropout(0.5)
        self.bidirectional = BIDIRECTIONAL
        self.append_bigram = USE_BIGRAM
        self.hidden = self.init_hidden()
        # if self.append_bigram:
        #     self.hidden2tag = nn.Linear(hidden_dim + bigram_size, tagset_size)
        # else:
        #     self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        if self.USE_CRF:
            self.crf = CRF(tagset_size, tag_to_ix, self.use_gpu)
            if self.use_gpu:
                self.crf = self.crf.cuda()

        if self.bidirectional:
            self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)
            b = np.sqrt(6.0 / (self.hidden2tag.weight.size(0) + self.hidden2tag.weight.size(1)))
            nn.init.uniform_(self.hidden2tag.weight, -b, b)
            self.hidden2tag.bias.data.zero_()
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # import pdb; pdb.set_trace()

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
                return (autograd.Variable(torch.randn(2, 1, self.hidden_dim )),
                        autograd.Variable(torch.randn(2, 1, self.hidden_dim )))

    def forward_lstm(self, sentence, chars, caps, feats, drop_prob):
        d = nn.Dropout(p=drop_prob)
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence)  # shape seq_length * emb_size
        # embeds = self.word_embeddings(sentence)  # shape seq_length * emb_size
        cap_embedding = self.cap_embeds(caps)
        ## seq feats
        student_embedding = self.student_embeds(feats['user'])
        country_embedding = self.country_embeds(feats['country'])
        days_embedding = self.days_embeds(feats['days'])
        client_embedding = self.client_embeds(feats['client'])
        session_embedding = self.session_embeds(feats['session'])
        format_embedding = self.format_embeds(feats['format'])
        time_embedding = self.time_embeds(feats['time'])
        ## token feats
        pos_embedding = self.pos_embeds(feats['pos'])
        edge_labels_embedding = self.edge_labels_embeds(feats['edge_labels'])
        edge_heads_embedding = self.edge_heads_embeds(feats['edge_heads'])

        if self.CNN == True:
            chars_embeds = self.char_embeds(chars).unsqueeze(1)
            cnn_output = self.char_cnn(d(chars_embeds))

            chars_embeds = nn.functional.max_pool2d(cnn_output,
                                                 kernel_size=(cnn_output.size(2), 1)).view(cnn_output.size(0), self.char_lstm_dim)
            if self.duolingo_student:
                if self.use_gpu:
                    embeds = torch.cat((embeds, chars_embeds, cap_embedding, student_embedding, country_embedding, days_embedding, client_embedding, session_embedding, format_embedding, time_embedding, pos_embedding, edge_labels_embedding, edge_heads_embedding), 1).cuda()
                else:
                    embeds = torch.cat((embeds, chars_embeds, cap_embedding, student_embedding, country_embedding, days_embedding, client_embedding, session_embedding, format_embedding, time_embedding, pos_embedding, edge_labels_embedding, edge_heads_embedding), 1)
            else:
                if self.use_gpu:
                    embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1).cuda()
                else:
                    embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
        else:
            if self.duolingo_student:
                if self.use_gpu:
                    embeds = torch.cat((embeds, cap_embedding, student_embedding, country_embedding, days_embedding, client_embedding, session_embedding, format_embedding, time_embedding, pos_embedding, edge_labels_embedding, edge_heads_embedding), 1).cuda()
                else:
                    embeds = torch.cat((embeds, cap_embedding, student_embedding, country_embedding, days_embedding, client_embedding, session_embedding, format_embedding, time_embedding, pos_embedding, edge_labels_embedding, edge_heads_embedding), 1)
            else:
                if self.use_gpu:
                    embeds = torch.cat((embeds, cap_embedding), 1).cuda()
                else:
                    embeds = torch.cat((embeds, cap_embedding), 1)
        # print(embeds.view(len(sentence), 1, -1).shape)
        #lstm_out, self.hidden = self.lstm(embeds.unsqueeze(1), self.hidden)
        lstm_out, _ = self.lstm(d(embeds).unsqueeze(1))
        # lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        lstm_out = d(lstm_out.view(len(sentence), self.hidden_dim*2))
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        # print("original shape before MLP "+str(lstm_out.view(len(sentence), -1).shape))
        # print("shape of onehot bigram "+str(bigram_one_hot))
        if self.append_bigram:
            # print("concatenated vector"+str(lstm_out.view(len(sentence), -1)))
            tag_space=self.hidden2tag(torch.cat([lstm_out.view(len(sentence), -1),bigram_one_hot],dim=1))
        else:
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores


    def neg_ll_loss(self, sentence, gold_labels, chars, caps, feats, drop_prob):
        lstm_feats = self.forward_lstm(sentence, chars, caps, feats, drop_prob)
        return self.crf.neg_ll_loss(sentence, gold_labels, lstm_feats)

    def forward(self, sentence, chars, caps, feats, drop_prob):
        if self.USE_CRF:
            feats = self.forward_lstm(sentence, chars, caps, feats, drop_prob)
            score, tag_seq = self.crf.forward(sentence, feats)
            return score, tag_seq
        else:
            scores = self.forward_lstm(sentence, chars, caps, feats, drop_prob)
            return scores
