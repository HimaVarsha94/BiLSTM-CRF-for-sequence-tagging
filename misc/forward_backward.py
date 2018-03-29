import numpy as np

# CRF transition matrix
# TODO: try init randomly as well, see which works best
self.trans_mat = np.zeros((tagset_size, tagset_size))

def forward_backward(self, state_probs, sent_len):
    c = np.zeros(sent_len)
    alpha = np.zeros((sent_len, self.tagset_size))
    beta = np.zeros((sent_len, self.tagset_size))
    epsilon = np.zeros((sent_len, self.tagset_size, self.tagset_size))

    # alphas
    for t in range(sent_len):
        for i in range(self.tagset_size):
            if t == 0:
                alpha[t][i] = 0.0
            else:
                for j in range(self.tagset_size):
                    alpha[t][i] += alpha[t - 1][j] * self.trans_mat[i][j]
                alpha[t][i] *= state_probs[t][i]

        alpha_sum = 0.0
        for i in range(self.tagset_size):
            alpha_sum += alpha[t][i]
        c[t] = 1.0 / alpha_sum
        for i in range(self.tagset_size):
            alpha[t][i] /= alpha_sum

    # betas
    for t in range(sent_len - 1, -1, -1):
        for i in range(self.tagset_size):
            if t == sent_len - 1:
                beta[0][i] = 1.0
            else:
                for j in range(self.tagset_size):
                    beta[t][i] += beta[t + 1][j] * self.trans_mat[j][i]
                ## lolol same prob (t+1 or t)?
                beta[t][i] *= state_probs[t + 1][i]

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
                    epsilon[t][i][j] = alpha[t - 1][j] * self.trans_mat[i][j] * state_probs[t][j] * beta[t][i]

    return epsilon

def update_crf(self, epsilon, sent_len):
    # update transition matrix
    for i in range(self.tagset_size):
        for t in range(sent_len):
            for j in range(self.tagset_size):
                self.trans_mat[i][j] += epsilon[t][i][j]
