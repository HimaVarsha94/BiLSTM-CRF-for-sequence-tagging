import torch
import torch.autograd as autograd
import torch.nn as nn

START_TAG = '<START>'
END_TAG = '<END>'
MIN_VAL = -100000

## accepts torch.Tensor of size NxN and returns log sum exp across second dimension
## OR accepts torch.Tensor of size 1xN and does same
def lse(vec):
    if len(vec.shape) == 2:
        max_vals = torch.max(vec,1)[0]
        max_vals_broadcasted = max_vals.repeat(vec.shape[0],1).transpose(0,-1)
        return torch.log(torch.sum(torch.exp(vec - max_vals_broadcasted),1)) + max_vals
    else:
        max_vals = torch.max(vec)
        return torch.log(torch.sum(torch.exp(vec - max_vals))) + max_vals

class CRF(nn.Module):

    def __init__(self, num_tags, tag_to_ix, use_gpu):
        super(CRF, self).__init__()
        # include start and end tags
        self.use_gpu = use_gpu
        self.num_tags = num_tags
        self.tag_to_ix = tag_to_ix

        ## init transition matrix
        init_trans = torch.zeros(num_tags, num_tags)
        # set transitions that should never happen to MIN_VAL
        init_trans[self.tag_to_ix[START_TAG],:] = MIN_VAL
        init_trans[:,self.tag_to_ix[END_TAG]] = MIN_VAL
        if self.use_gpu:
            init_trans = init_trans.cuda()
        self.trans = nn.Parameter(init_trans)

    def forward_alg(self, feats):
        # init alphas
        if self.use_gpu:
            init_alpha = torch.cuda.FloatTensor(1,self.num_tags).fill_(MIN_VAL)
        else:
            init_alpha = torch.FloatTensor(1,self.num_tags).fill_(MIN_VAL)

        init_alpha[0,self.tag_to_ix[START_TAG]] = 0

        alpha = autograd.Variable(init_alpha)
        # loop over sentence
        for feat in feats:
            # TODO: try adding feat inside
            alpha = lse(self.trans + alpha) +  feat

        final_alpha = alpha + self.trans[self.tag_to_ix[END_TAG]]
        return lse(final_alpha)

    def viterbi(self, feats, sent_len):
        ## init table of probs of most likely path so far
        if self.use_gpu:
            init_v_probs = torch.cuda.FloatTensor(1,self.num_tags).fill_(MIN_VAL)
        else:
            init_v_probs = torch.FloatTensor(1,self.num_tags).fill_(MIN_VAL)

        init_v_probs[0,self.tag_to_ix[START_TAG]] = 0
        vprobs = autograd.Variable(init_v_probs)

        backptrs = []

        ## fill out table
        for feat in feats:
            # TODO: try adding feats now rather than later, why does it matter?
            maxes, argmax = torch.max(self.trans + vprobs, 1)
            backptrs.append(argmax.cpu().data.numpy().tolist())
            vprobs = (maxes+feat).view(1,-1)

        ## transition to END_TAG
        temp_probs = self.trans[self.tag_to_ix[END_TAG]] + vprobs
        final_tag = int(torch.max(temp_probs,1)[1])
        final_score = temp_probs[0][final_tag]

        ## follow backpointers to get best sequence
        best_seq = [final_tag]
        tag = final_tag
        for backptr_vec in reversed(backptrs):
            best_seq.append(backptr_vec[best_seq[-1]])
        ## don't want to return START_TAG
        best_seq.reverse()

        return final_score, best_seq[1:]

    def get_gold_scores(self, feats, tags):
        if self.use_gpu:
            score = autograd.Variable(torch.cuda.FloatTensor([0]))
        else:
            score = autograd.Variable(torch.FloatTensor([0]))

        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i,feat in enumerate(feats):
            score += self.trans[tags[i+1],tags[i]] + feat[tags[i+1]]
        # end tag
        score += self.trans[self.tag_to_ix[END_TAG],tags[-1]]
        return score

    def neg_ll_loss(self, sentence, gold_labels, feats):
        forward_scores = self.forward_alg(feats)
        gold_scores = self.get_gold_scores(feats, gold_labels)
        return forward_scores - gold_scores

    def forward(self, sentence, feats):
        score, tag_seq = self.viterbi(feats, len(sentence))
        return score, tag_seq
