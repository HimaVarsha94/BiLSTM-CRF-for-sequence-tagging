import nltk
import re
import numpy as np, pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm import LSTMTagger
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import FeatureHasher

torch.manual_seed(1)


def get_embeddings_matrix(data, USE_BIGRAM, USE_HASHING=False, HASHING_SIZE=20):
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

    bigram_to_ix = {}
    if USE_BIGRAM and not USE_HASHING:

        for sent in data:

            bi_grams = nltk.bigrams(["<s>"] + sent)

            # bigrams added to the dictionary
            for grams in bi_grams:
                bigram = grams[0] + '|' + grams[1]
                if bigram not in bigram_to_ix:
                    bigram_to_ix[bigram] = len(bigram_to_ix)

        with open('./chunking_models/bigram_to_ix.pkl', 'wb') as f:
            pickle.dump(bigram_to_ix, f)
        print("total bigram size is " + str(len(bigram_to_ix)))
    if USE_HASHING:
        return embeddings_mat, word_to_ix, bigram_to_ix, HASHING_SIZE

    return embeddings_mat, word_to_ix, bigram_to_ix, len(bigram_to_ix)


def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        w = w.lower()
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def prepare_bigram(seq, bigram_to_ix, bigram_size, USE_HASHING=False, vectorizer=None):
    # tensor=torch.LongTensor(len(seq)).zero_()
    values = []
    bi_grams = nltk.bigrams(["<s>"] + seq)
    for grams in bi_grams:
        bigram = grams[0] + '|' + grams[1]
        if not USE_HASHING:
            value = np.zeros(bigram_size)
            if bigram in bigram_to_ix:
                value[bigram_to_ix[bigram]] = 1
            values.append(value)
        else:
            values.append(vectorizer.transform([{bigram: 1}]).toarray())
    tensor = torch.FloatTensor(values).squeeze(1)
    # print("bigram feature vector " + str(tensor.shape))
    # print("does it matches sequence length " + str(len(seq)))
    return autograd.Variable(tensor)


def prepare_spelling(seq, SPELLING_SIZE=7):
    values = []

    for w in seq:
        value=np.zeros(SPELLING_SIZE)
        for i in range(SPELLING_SIZE):
            if i==0:
                value[i]=(1 if (w[0]>='A' and w[0]<='Z') else 0)
                continue
            elif i==1:
                value[i]=(1 if w.upper()==w else 0)
                continue
            elif i==2:
                value[i]=(1 if w.lower()==w else 0)
                continue
            elif i==3:
                value[i]=(1 if w[1:].upper()!=w[1:] else 0)
                continue
            elif i==4:
                value[i]=(1 if any(char.isdigit() for char in w) else 0)
                continue
            elif i==5:
                value[i]=(0 if re.match("^[a-zA-Z0-9]*$", w) else 1)
                continue
            elif i==6:
                value[i]=(0 if w.replace("'", "")==w else 1)
        values.append(value)
    tensor = torch.FloatTensor(values).squeeze(1)
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
    print(counter)
    print("Example line of training data and Y\n\n" + str(X[0]) + " \n\n" + str(y[0]) + "\n")
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


def main():
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 300  # the dimension for single direction
    USE_CRF = False
    BIDIRECTIONAL = True
    USE_BIGRAM = True
    USE_HASHING = True
    HASHING_SIZE = 20

    USE_SPELLING = True
    SPELLING_SIZE = 7

    if USE_HASHING:
        vectorizer = FeatureHasher(n_features=HASHING_SIZE)

    training_data, y = load_chunking(train=True)
    test_X, test_y = load_chunking(test=True)
    emb_mat, word_to_ix, bigram_to_ix, bigram_size = get_embeddings_matrix(training_data, USE_BIGRAM, USE_HASHING,
                                                                           HASHING_SIZE)

    tag_to_ix = tag_indices(y)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), emb_mat, USE_CRF, BIDIRECTIONAL,
                       USE_BIGRAM, bigram_size, USE_SPELLING, SPELLING_SIZE)

    loss_function = nn.NLLLoss()
    parameters = model.parameters()
    # parameters = filter(lambda p: model.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=0.1)

    len_train = len(training_data)
    len_test = len(test_X)
    for epoch in range(50):
        loss_cal = 0.0
        print(epoch)
        for ind in range(len_train):
            sentence = training_data[ind]
            # print("sentence sequence len"+str(len(sentence)))
            tags = y[ind]
            model.zero_grad()
            # check this
            model.hidden = model.init_hidden()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            bigram_one_hot = None

            if USE_BIGRAM:
                bigram_one_hot = prepare_bigram(sentence, bigram_to_ix, bigram_size, USE_HASHING, vectorizer)

            spelling_one_hot = None

            if USE_SPELLING:
                spelling_one_hot = prepare_spelling(sentence, SPELLING_SIZE)

            tag_scores = model(sentence_in, bigram_one_hot, spelling_one_hot)

            loss = loss_function(tag_scores, targets)
            loss_cal += loss
            loss.backward()

            optimizer.step()
        PATH = './chunking_models/model_epoch' + str(epoch)
        torch.save(model.state_dict(), PATH)
        model.load_state_dict(torch.load(PATH))

        print(loss_cal)
        print("Testing!!")
        correct = 0
        total = 0
        for ind in range(len_test):
            sentence = test_X[ind]
            tags = test_y[ind]
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            bigram_one_hot = None

            if USE_BIGRAM:
                bigram_one_hot = prepare_bigram(sentence, bigram_to_ix, bigram_size, USE_HASHING, vectorizer)

            spelling_one_hot = None

            if USE_SPELLING:
                spelling_one_hot = prepare_spelling(sentence)

            tag_scores = model(sentence_in, bigram_one_hot, spelling_one_hot)

            prob, predicted = torch.max(tag_scores.data, 1)
            correct += (predicted == targets.data).sum()
            total += targets.size(0)
            loss = loss_function(tag_scores, targets)
        print("Accuracy of epoch {} is {}".format(epoch, float(correct) / total))


if __name__ == '__main__':
    main()
