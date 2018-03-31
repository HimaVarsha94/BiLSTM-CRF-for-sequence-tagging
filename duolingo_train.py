import numpy as np, pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.lstm import LSTMTagger
# import baseline
torch.manual_seed(1)

def get_embeddings_matrix(data):
    word_to_ix = {'unk':0}
    for sent in data:
        for word in sent:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    with open('./chunking_models/word_to_ix.pkl', 'wb') as f:
        pickle.dump(word_to_ix, f)
        
    #this contains a list of all senna obj embeddings
    with open('./senna_embeddings/senna_obj.pkl', 'rb') as f:
        senna_obj = pickle.load(f)
    #embeddings matrix with each row corresponding to a word to pass to nn.embedding layer
    embeddings_mat = np.zeros((len(word_to_ix), 50))

    for word in word_to_ix:
        if word in senna_obj:
            embeddings_mat[word_to_ix[word]] = senna_obj[word]
        else:
            embeddings_mat[word_to_ix[word]] = np.zeros(50)

    with open('./chunking_models/conll2000_matrix', 'wb') as f:
        pickle.dump(embeddings_mat, f)

    return embeddings_mat, word_to_ix


def prepare_sequence(seq, to_ix, tag=False):
    idxs = []
    for w in seq:
        if tag==False:
            w = w.lower()
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
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
    return X, y

def load_duolingo(train=False, test=False):
    with open('./data/duolingo/data.txt', 'rb') as f:
        all_data = pickle.load(f)
        print("all_data_length "+str(len(all_data)))
    with open('./data/duolingo/data_labels.txt', 'rb') as f:
        all_data_labels = pickle.load(f)
    length = len(all_data)
    split = int(length*0.9)
    if train == True:
        return all_data[:split], all_data_labels[:split]
    if test == True:
        pickle.dump(all_data[split:],open("./data/duolingo/test_data.txt","wb"))
        pickle.dump(all_data_labels[split:], open("./data/duolingo/test_data_label.txt", "wb"))
        print("testing data length is "+str(len(all_data_labels[split:])))
        return all_data[split:], all_data_labels[split:]
    import pdb; pdb.set_trace()

def tag_indices(X, y):
    tag_to_idx = {}
    for sent_tag in y:
        for tag in sent_tag:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)

    with open('./chunking_models/tag_to_idx', 'wb') as f:
        pickle.dump(tag_to_idx, f)
    return tag_to_idx

def main():
    training_data, y = load_duolingo(train=True)
    from chunking_train import avg_len
    print("average len is " + str(avg_len(training_data)))
    test_X, test_y = load_duolingo(test=True)
    emb_mat, word_to_ix = get_embeddings_matrix(training_data)
    tag_to_ix = tag_indices(training_data, y) 
    

    EMBEDDING_DIM = 50
    HIDDEN_DIM = 300

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), emb_mat)
    loss_function = nn.NLLLoss()
    parameters = model.parameters()
    # parameters = filter(lambda p: model.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=0.1)

    len_train = len(training_data)
    len_test = len(test_X)
    for epoch in range(50):
        loss_cal = 0.0
        print(epoch)
        for ind in range(int(len_train/100)):
            if ind%1000==0:
                print(ind)
            sentence = training_data[ind]
            tags = y[ind]
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix, tag=True)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss_cal += loss
            loss.backward()

            optimizer.step()
        PATH = './duolingo_models/model_epoch'+str(epoch)
        torch.save(model.state_dict(), PATH)
        model.load_state_dict(torch.load(PATH))

        print(loss_cal)
        print("Testing!!")
        correct = 0
        total = 0
        predicted_values = []
        print(len_test)
        for ind in range(len_test):
            if ind%5000==0:
                print(ind)
            sentence = test_X[ind]
            tags = test_y[ind]
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix, tag=True)
            tag_scores = model(sentence_in)
            prob, predicted = torch.max(tag_scores.data, 1)
            predicted_values.append(predicted)
            correct += (predicted == targets.data).sum()
            total += targets.size(0)
            loss = loss_function(tag_scores, targets)
        print("Accuracy of epoch {} is {}".format(epoch, float(correct)/total))
        with open('./duolingo_models/results'+str(epoch), 'wb') as f:
            pickle.dump(predicted_values, f)


if __name__ == '__main__':
    main()