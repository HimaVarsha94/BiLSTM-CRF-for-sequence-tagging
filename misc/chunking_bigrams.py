import numpy as np, pickle, nltk
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm import LSTMTagger
torch.manual_seed(1)

def get_embeddings_matrix(data):
    word_to_ix = {'unk':0}

    for sent in data:
        sent = [word.lower() for word in sent]
        bi_grams = nltk.bigrams(sent)

        #bigrams added to the dictionary
        for gram in bi_grams:
            word = gram[0]+'|'+gram[1]
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

        if sent[-1] not in word_to_ix:
                word_to_ix[sent[-1]] = len(word_to_ix)
        
    with open('./chunking_models/word_to_ix.pkl', 'wb') as f:
        pickle.dump(word_to_ix, f)
        
    #this contains a list of all senna obj embeddings
    with open('./senna_embeddings/senna_obj.pkl', 'rb') as f:
        senna_obj = pickle.load(f)
    #embeddings matrix with each row corresponding to a word to pass to nn.embedding layer
    embeddings_mat = np.zeros((len(word_to_ix), 150))

    for bigrams in word_to_ix:
        if bigrams == 'unk':
            bigram_emb = np.zeros(150)
            embeddings_mat[word_to_ix['unk']] = bigram_emb
            continue
        words = bigrams.split('|')
        if words[0] in senna_obj:
            emb0 = senna_obj[words[0]]
        else:
            emb0 = np.zeros(50)
        if len(words) == 1:

            bigram_emb = np.append(np.append(emb0, emb0), np.zeros(50))
            embeddings_mat[word_to_ix[words[0]]] = bigram_emb
        else:
            #check if next gram exists in senna
            if words[1] in senna_obj:
                emb1 = senna_obj[words[1]]
            else:
                emb1 = np.zeros(50)
            bigram_emb = np.append(np.append(emb0, emb0), emb1)
            embeddings_mat[word_to_ix[bigrams]] = bigram_emb

    with open('./chunking_models/conll2000_matrix', 'wb') as f:
        pickle.dump(embeddings_mat, f)

    return embeddings_mat, word_to_ix

def prepare_sequence(sent, to_ix):
    idxs = []
    sent = [word.lower() for word in sent]
    bi_grams = nltk.bigrams(sent)
    for gram in bi_grams:
        word = gram[0]+'|'+gram[1]
        if word not in to_ix:
            print("Ouch ", word)
            idxs.append(0)
        else:
            idxs.append(to_ix[word])
    if sent[-1] not in to_ix:
        print("last Ouch ", sent[-1])
        idxs.append(0)
    else:
        idxs.append(to_ix[sent[-1]])

    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def prepare_tag(seq, to_ix):
    idxs = []
    for w in seq:
        w = w.lower()
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    # idxs = [to_ix[w.lower()] for w in seq]
    tensor = torch.LongTensor(idxs)
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
    sys.exit(0)

def tag_indices(X, y):
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
    training_data, y = load_chunking(train=True)
    test_X, test_y = load_chunking(test=True)
    emb_mat, word_to_ix = get_embeddings_matrix(training_data)
    tag_to_ix = tag_indices(training_data, y) 
    

    EMBEDDING_DIM = 150
    HIDDEN_DIM = 300

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), None)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    len_train = len(training_data)
    len_test = len(test_X)
    for epoch in range(50):
        loss_cal = 0.0
        print(epoch)
        for ind in range(len_train):
            sentence = training_data[ind]
            tags = y[ind]
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_tag(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss_cal += loss
            loss.backward()
            optimizer.step()
        PATH = './chunking_models/model_epoch'+str(epoch)
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
            targets = prepare_tag(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            prob, predicted = torch.max(tag_scores.data, 1)
            correct += (predicted == targets.data).sum()
            total += targets.size(0)
            loss = loss_function(tag_scores, targets)
        print("Accuracy of epoch {} is {}".format(epoch, float(correct)/total))


if __name__ == '__main__':
    main()