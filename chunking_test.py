import pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm import LSTMTagger
torch.manual_seed(1)
from sequence_models_tutorial import chunking_preprocess, load_chunking

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def word_indices():
    with open('./chunking_models/tag_to_idx', 'rb') as f:
        tag_to_idx = pickle.load(f)
    with open('./chunking_models/word_to_ix.pkl', 'rb') as f:
        word_to_ix = pickle.load(f)
    # with open('./chunking_models/conll2000_matrix', 'rb') as f:
    #     embeddings_mat = pickle.load(f)
    return word_to_ix, tag_to_idx#, embeddings_mat

def main():
    test_X, y = load_chunking(train=False, test=True)
    word_to_ix, tag_to_ix = word_indices()

    EMBEDDING_DIM = 50
    HIDDEN_DIM = 6
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), None)
    loss_function = nn.NLLLoss()

    len_test = len(test_X)
    for epoch in range(10):
        correct = 0
        total = 0
        print("Epoch number is ", epoch)
        PATH = './chunking_models/model_epoch'+str(epoch)
        model.load_state_dict(torch.load(PATH))
        for ind in range(len_test):
            sentence = test_X[ind]
            tags = y[ind]
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            prob, predicted = torch.max(tag_scores.data, 1)
            correct += (predicted == targets.data).sum()
            total += targets.size(0)
            loss = loss_function(tag_scores, targets)
        print("Accuracy of epoch {} is {}".format(epoch, float(correct)/total))

if __name__ == '__main__':
    main()