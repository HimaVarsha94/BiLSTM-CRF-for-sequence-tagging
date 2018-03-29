import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

START_TAG = -1
END_TAG = -2
MIN_VAL = -100000

class CRF(nn.Module):

    def __init__(self, num_tags, gpu):
        super(CRF, self).__init__()
        # include start and end tags
        self.gpu = gpu
        self.num_tags = num_tags

        ## init transition matrix
        init_trans = torch.zeros(num_tags+2, num_tags+2)
        # set transitions that should never happen to MIN_VAL
        init_trans[START_TAG,:] = MIN_VAL
        init_trans[:,END_TAG] = MIN_VAL
        if gpu:
            init_trans = init_trans.cuda
        self.trans = nn.Parameter(init_trans)

    # not sure i need this
    def init_hidden(self):
        pass

    def forward_alg(self, feats):
        # init alphas
        init_alpha = torch.Tensor(1,self.num_tags+2).fill_(MIN_VAL)
        init_alpha[0,START_TAG] = 0
        alpha = autograd.Variable(init_alpha)
        if gpu:
            alpha = alpha.cuda()

        # loop over sentence
        for feat in feats:
            inner_alpha = torch.zeros(self.num_tags)
            # for each tag transition calculate alpha scores
            for to_tag in range(self.num_tags+2):
                # add transition to last alpha score
                to_sum = self.trans[to_tag].view(1 ,-1) + alpha
                # calc log sum exp and add emission probability
                inner_alpha[to_tag] = lsm(to_sum) + feat[to_tag].view(1,-1)
            alpha = inner_alpha.view(1,-1)
        # add final transition score
        final_alpha = alpha + self.trans[END_TAG]
        return lsm(final_alpha)

    def viterbi(self, feats):
        pass

    def get_gold_scores(self, feats, tags):
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.IntTensor([START_TAG]), tags])
        for i,feat in enumerate(feats):
            score += self.trans[tag[i+1],tags[i]] + feat[tags[i+1]]
        # end tag
        score += self.trans[END_TAG,tags[-1]]
        return score

    def neg_ll_loss(self, sentence, gold_labels, feats):
        forward_scores = forward_alg(feats)
        gold_scores = get_gold_scores(feats, gold_labels)
        return forward_scores - gold_scores

    def forward(self, sentence, feats):
        score, tag_seq = viterbi(feats)
        return score, tag_seq
