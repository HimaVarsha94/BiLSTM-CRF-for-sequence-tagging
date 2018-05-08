from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

# import lib

from constant import PAD


class Dataset(object):
    def __init__(self, data, labels, cuda, batch_size, eval=True, session_feats=None):
        self.words = [torch.LongTensor(feat['words']) for feat in data]
        self.pos = [torch.LongTensor(feat['pos']) for feat in data]
        self.edge_labels = [torch.LongTensor(feat['edge_labels']) for feat in data]
        self.edge_heads = [torch.LongTensor(feat['edge_heads']) for feat in data]
        # self.chars = [torch.LongTensor(feat['chars']) for feat in data]
        self.labels = [torch.LongTensor(feat) for feat in labels]

        if session_feats is not None:
            self.use_user = True
            self.use_format = True
            self.users = [torch.LongTensor([feat['user']]) for feat in session_feats]
            self.countries = [torch.LongTensor([feat['country']]) for feat in session_feats]
            self.days = [torch.LongTensor([feat['days']]) for feat in session_feats]
            self.data_format = [torch.LongTensor([feat['format']]) for feat in session_feats]
            self.session = [torch.LongTensor([feat['session']]) for feat in session_feats]
            self.client = [torch.LongTensor([feat['client']]) for feat in session_feats]
            self.time = [torch.LongTensor([feat['time']]) for feat in session_feats]
            # input(str(self.days))
        else:
            self.use_user = False
            self.use_format = False
            self.users = range(len(self.words))
            self.countries = range(len(self.words))
            self.days = range(len(self.words))
            self.data_format = range(len(self.words))
            self.session = range(len(self.words))
            self.client = range(len(self.words))
            self.time = range(len(self.words))

        self.cuda = cuda
        self.eval = eval

        self.batchSize = batch_size
        self.numBatches = math.ceil(len(self.words) / self.batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False, pad=0):  # here data is a sequence of LongTensor
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(pad)  # create a new tensor and fill in PAD value
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)

        wordsBatch, lengths = self._batchify(self.words[index * self.batchSize:(index + 1) * self.batchSize],
                                             include_lengths=True)

        posBatch = self._batchify(self.pos[index * self.batchSize:(index + 1) * self.batchSize])

        # Note: tgtBatch length list could be different from above lengths variable, although their correlation could be high,
        # e.g. a german sentence of length 24 is likely to have an english sentence of length 25

        edgeBatch = self._batchify(self.edge_labels[index * self.batchSize:(index + 1) * self.batchSize])

        labelBatch = self._batchify(self.labels[index * self.batchSize:(index + 1) * self.batchSize],
                                    pad=-1)  # batch_size * seq_len

        if self.use_user:
            # print("creating batch " + str(self.users[index * self.batchSize:(index + 1) * self.batchSize]))
            usersBatch = self.users[index * self.batchSize:(index + 1) * self.batchSize]
            countriesBatch = self.countries[index * self.batchSize:(index + 1) * self.batchSize]
            daysBatch = self.days[index * self.batchSize:(index + 1) * self.batchSize]


        else:
            usersBatch, countriesBatch, daysBatch = len(wordsBatch), len(wordsBatch), len(wordsBatch)
        if self.use_format:
            formatBatch = self.data_format[index * self.batchSize:(index + 1) * self.batchSize]
            sessionBatch = self.session[index * self.batchSize:(index + 1) * self.batchSize]
            clientBatch = self.client[index * self.batchSize:(index + 1) * self.batchSize]
            timeBatch = self.time[index * self.batchSize:(index + 1) * self.batchSize]
        else:
            formatBatch, sessionBatch, clientBatch, timeBatch = len(wordsBatch), len(wordsBatch), len(wordsBatch), len(wordsBatch)

        # within batch sort by decreasing length.
        indices = range(len(wordsBatch))
        batch = zip(indices, wordsBatch, posBatch, edgeBatch, labelBatch, usersBatch, countriesBatch, daysBatch, formatBatch, sessionBatch, clientBatch, timeBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, wordsBatch, posBatch, edgeBatch, labelBatch, usersBatch, countriesBatch, daysBatch, formatBatch, sessionBatch, clientBatch, timeBatch = zip(*batch)

        def wrap(b):
            b = torch.stack(b, 0).t().contiguous()  # everything was transposed to have seqlen * batch_size

            # contiguous() pytorch 里面的 contiguous()是以 C 为顺序保存在内存里面，如果不是，则返回一个以 C 为顺序保存的tensor. A contiguous array is just an array stored in an unbroken block of memory:
            # originally b could be batch_size * seq_len .stack() means concatenates sequence of tensors b along  dimension 0; t() means: input to be a matrix (2-D tensor) and transposes dimensions 0 and 1.

            if self.cuda:
                b = b.cuda()
            b = Variable(
                b,
                volatile=self.eval)  # set the input to a network to volatile if you are doing inference only and won't be running backpropagation in order to conserve memory.
            return b

        return (wrap(wordsBatch), lengths), wrap(posBatch), wrap(edgeBatch), wrap(labelBatch), indices, (wrap(usersBatch) if self.use_user else None), (wrap(countriesBatch) if self.use_user else None), (
                   wrap(daysBatch) if self.use_user else None), (wrap(formatBatch) if self.use_format else None), (wrap(sessionBatch) if self.use_format else None), (wrap(clientBatch) if self.use_format else None), (wrap(timeBatch) if self.use_format else None)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(
            zip(self.words, self.pos, self.edge_labels, self.edge_heads, self.labels, self.users, self.countries,
                self.days, self.data_format, self.session, self.client, self.time))
        random.shuffle(data)
        self.words, self.pos, self.edge_labels, self.edge_heads, self.labels, self.users, self.countries, self.days, self.data_format, self.session, self.client, self.time = zip(*data)

    # def restore_pos(self, sents):
    #     sorted_sents = [None] * len(self.pos)
    #     for sent, idx in zip(sents, self.pos):
    #         sorted_sents[idx] = sent
    #     return sorted_sents
