import numpy as np, pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.lstm import LSTMTagger
from models.lstm_cnn import BILSTM_CNN
from sklearn.metrics import f1_score, precision_score, recall_score

torch.manual_seed(1)
use_gpu = 1

def get_embeddings_matrix(data):
    word_to_ix = {'unk': 0}
    for sent in data:
        for word in sent:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    with open('./chunking_models/word_to_ix.pkl', 'wb') as f:
        pickle.dump(word_to_ix, f)

    # this contains a list of all senna obj embeddings
    with open('./senna_embeddings/senna_obj.pkl', 'rb') as f:
        senna_obj = pickle.load(f)
    # embeddings matrix with each row corresponding to a word to pass to nn.embedding layer
    embeddings_mat = np.zeros((len(word_to_ix), 50))

    for word in word_to_ix:
        if word in senna_obj:
            embeddings_mat[word_to_ix[word]] = senna_obj[word]
        else:
            embeddings_mat[word_to_ix[word]] = np.zeros(50)

    with open('./chunking_models/conll2000_matrix', 'wb') as f:
        pickle.dump(embeddings_mat, f)

    print("shape of emb_mat " + str(embeddings_mat.shape) + " word_to_ix is a dict of size " + str(
        len(word_to_ix.keys())))

    return embeddings_mat, word_to_ix


def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        w = w.lower()
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    tensor = torch.cuda.LongTensor(idxs)
    return autograd.Variable(tensor)


def chunking_preprocess(datafile, senna=True):
    counter = 0
    data = []
    X = []
    y = []
    new_data = []
    new_label = []
    for line in datafile:
        counter += 1
        line = line.strip()
        tokens = line.split(' ')
        if len(tokens) == 1:
            X.append(new_data)
            y.append(new_label)
            new_data = []
            new_label = []
        else:
            new_data.append(tokens[0])
            new_label.append(tokens[2])
    # print(counter)
    return X, y


def load_chunking(train=False, test=False):
    if train == True:
        train_data = open('./data/conll2000/train.txt')
        X_train, y_train = chunking_preprocess(train_data)
        return X_train, y_train
    if test == True:
        print("testing data..")
        test_data = open('./data/conll2000/test.txt')
        X_test, y_test = chunking_preprocess(test_data)
        return X_test, y_test
    import pdb;
    pdb.set_trace()


def tag_indices(y):
    tag_to_idx = {}
    for sent_tag in y:
        for tag in sent_tag:
            tag = tag.lower()
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)

    with open('./chunking_models/tag_to_idx', 'wb') as f:
        pickle.dump(tag_to_idx, f)
    return tag_to_idx


def char_dict(data):
    char_to_ix = {}
    for sent in data:
        for word in sent:
            word = word.lower()
            for character in word:
                if character not in char_to_ix:
                    char_to_ix[character] = len(char_to_ix)

    with open('./chunking_models/char_to_ix.pkl', 'wb') as f:
        pickle.dump(char_to_ix, f)

    return char_to_ix

def prepare_words(sentence, char_to_ix):
    d = []
    for w in sentence:
        w = w.lower()
        idxs = []
        for char in w:
            if char in char_to_ix:
                idxs.append(char_to_ix[char])
            else:
                idxs.append(0)
        d.append(idxs)
    return d
    # tensor = torch.LongTensor(d)
    # return autograd.Variable(tensor)

def char_emb(chars2):
    chars2_length = [len(c) for c in chars2]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
    for i, c in enumerate(chars2):
        chars2_mask[i, :chars2_length[i]] = c
    chars2_mask = autograd.Variable(torch.cuda.LongTensor(chars2_mask))
    return chars2_mask


def avg_len(data):
    lens=[]
    for d in data:
        try:
            lens+=[len(d)]
        except:
            pass
    return np.mean(lens),np.std(lens)


def main():
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 300  # the dimension for single direction
    USE_CRF = True
    CNN = True
    batch_size = 2

    training_data, y = load_chunking(train=True)
    print("average len is "+str(avg_len(training_data)))
    test_X, test_y = load_chunking(test=True)
    emb_mat, word_to_ix = get_embeddings_matrix(training_data)
    tag_to_ix = tag_indices(y)
    char_to_ix = char_dict(training_data)

    if CNN == False:
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), emb_mat, USE_CRF)
    else:
        print("Cnn")
        model = BILSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, CNN=True)
        model = model.cuda()

    loss_function = nn.NLLLoss()
    parameters = model.parameters()
    # parameters = filter(lambda p: model.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=0.1)

    len_train = len(training_data)
    len_test = len(test_X)
    for epoch in range(50):
        loss_cal = 0.0
        print('Epoch: {}'.format(epoch))
        for ind in range(len_train):
            sentence = training_data[ind]
            tags = y[ind]
            model.zero_grad()
            # check this
            model.hidden = model.init_hidden()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            if CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                tag_scores = model(sentence_in,char_em)
            else:
                tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss_cal += loss
            loss.backward()

            optimizer.step()
        PATH = './chunking_models/model_epoch' + str(epoch)
        torch.save(model.state_dict(), PATH)
        model.load_state_dict(torch.load(PATH))

        print(loss_cal)
        print("Finished one epoch and Testing!!")
        correct = 0
        total = 0
        all_predicted = []
        all_targets= []

        for ind in range(len_test):
            sentence = test_X[ind]
            tags = test_y[ind]

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            if CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                tag_scores = model(sentence_in,char_em)
            else:
                tag_scores = model(sentence_in)
            prob, predicted = torch.max(tag_scores.data, 1)
            correct += (predicted == targets.data).sum()
            total += targets.size(0)

            if use_gpu:
                all_predicted = all_predicted + (autograd.Variable(predicted).data.cpu().numpy().tolist())
                all_targets = all_targets + (targets.data.cpu().numpy().tolist())
            else:
                all_predicted.append(predicted)
                all_targets.append(targets)


            # loss = loss_function(tag_scores, targets)
        print("F1 score weighted is ", f1_score(all_targets, all_predicted, average='weighted'))
        print("F1 score micro is ", f1_score(all_targets, all_predicted, average='micro'))
        print("Avg Precision ", precision_score(all_targets, all_predicted, average='weighted'))
        print("Avg Recall ", recall_score(all_targets, all_predicted, average='weighted'))
            # loss = loss_function(tag_scores, targets)
        print("Accuracy of epoch {} is {}".format(epoch, float(correct) / total))

if __name__ == '__main__':
    main()
