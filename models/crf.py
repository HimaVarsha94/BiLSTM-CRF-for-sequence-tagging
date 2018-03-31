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

    def __init__(self, num_tags, tag_to_ix, gpu):
        super(CRF, self).__init__()
        # include start and end tags
        self.gpu = gpu
        self.num_tags = num_tags
        self.tag_to_ix = tag_to_ix

        ## init transition matrix
        init_trans = torch.zeros(num_tags, num_tags)
        # set transitions that should never happen to MIN_VAL
        init_trans[self.tag_to_ix[START_TAG],:] = MIN_VAL
        init_trans[:,self.tag_to_ix[END_TAG]] = MIN_VAL
        if self.gpu:
            init_trans = init_trans.cuda()
        self.trans = nn.Parameter(init_trans)

    def forward_alg(self, feats):
        # init alphas
        init_alpha = torch.cuda.FloatTensor(1,self.num_tags).fill_(MIN_VAL)
        init_alpha[0,self.tag_to_ix[START_TAG]] = 0

        #if self.gpu:
            #init_alpha = init_alpha.cuda()
        alpha = autograd.Variable(init_alpha)
        # loop over sentence
        for feat in feats:
            # inner_alpha = []
            # for each tag transition calculate alpha scores
            # TODO: try adding feat inside
            alpha = lse(self.trans + alpha) +  feat
            # # for to_tag in range(self.num_tags):
            #     # add transition to last alpha score
            #     to_sum = self.trans[to_tag].view(1 ,-1) + alpha
            #     # calc log sum exp and add emission probability
            #     inner_alpha.append(lse(to_sum) + feat[to_tag].view(1,-1))
            # alpha = torch.cat(inner_alpha).view(1,-1)
        # add final transition score
        final_alpha = alpha + self.trans[self.tag_to_ix[END_TAG]]
        return lse(final_alpha)

    def viterbi(self, feats, sent_len):
        ## init table of probs of most likely path so far
        init_v_probs = torch.cuda.FloatTensor(1,self.num_tags).fill_(MIN_VAL)
        init_v_probs[0,self.tag_to_ix[START_TAG]] = 0
        vprobs = autograd.Variable(init_v_probs)
        #if self.gpu:
        #    vprobs = vprobs.cuda()
        ## init table of backpointers, same size
        backptrs = []

        ## fill out table
        for feat in feats:
            # TODO: try adding feats now
            maxes, argmax = torch.max(self.trans + vprobs, 1)
            backptrs.append(argmax.cpu().data.numpy().tolist())
            vprobs = (maxes+feat).view(1,-1)
            # inner_vprobs = []
            # inner_backptrs = []
            # for to_tag in range(self.num_tags):
            #     ## calc probs for transition to to_tag
            #     ## add in emission probs later (they don't depend on to_tag)
            #     temp_probs = self.trans[to_tag] + vprobs
            #     ## this is best last tag
            #     best_tag = torch.max(temp_probs,1)[1]
            #     inner_backptrs.append(best_tag.data[0])
            #     inner_vprobs.append(temp_probs[0][best_tag])
            # backptrs.append(inner_backptrs)
            # vprobs = (torch.cat(inner_vprobs) + feat).view(1,-1)

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
        score = autograd.Variable(torch.cuda.FloatTensor([0]))
        #if self.gpu:
        #    score = score.cuda()
        # print(type(tags))
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
