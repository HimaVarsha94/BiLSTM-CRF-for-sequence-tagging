import time, os
import numpy as np, pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from models.lstm import LSTMTagger
from models.lstm_cnn import BILSTM_CNN
# from models.bilstm_crf_cnn import BiLSTM_CRF_CNN
from sklearn.metrics import f1_score, precision_score, recall_score
from random import shuffle
torch.cuda.set_device(3)
# torch.manual_seed(1)
use_gpu = 0

START_TAG = '<START>'
END_TAG = '<END>'

def cap_feature(s):
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def get_embeddings_matrix(data):
    word_to_ix = {'unk':0}
    for sent in data:
        for word in sent:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    with open('./models/pos_models/word_to_ix.pkl', 'wb') as f:
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

    with open('./models/pos_models/pos_matrix', 'wb') as f:
        pickle.dump(embeddings_mat, f)

    return embeddings_mat, word_to_ix

def get_glove_matrix(data):
    word_to_ix = {'unk':0}
    for sent in data:
        for word in sent:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    with open('./models/pos_models/word_to_ix.pkl', 'wb') as f:
        pickle.dump(word_to_ix, f)
    #this contains a list of all senna obj embeddings
    with open('./senna_embeddings/glove_obj.pkl', 'rb') as f:
        glove_obj = pickle.load(f)
    #embeddings matrix with each row corresponding to a word to pass to nn.embedding layer
    embeddings_mat = np.zeros((len(word_to_ix), 100))

    for word in word_to_ix:
        if word in glove_obj:
            embeddings_mat[word_to_ix[word]] = glove_obj[word]
        else:
            embeddings_mat[word_to_ix[word]] = np.random.rand(100)

    with open('./models/pos_models/ner_matrix', 'wb') as f:
        pickle.dump(embeddings_mat, f)

    return embeddings_mat, word_to_ix

def prepare_sequence(seq, to_ix, tag=False):
    idxs = []
    for w in seq:
        if tag == False:
            w = w.lower()
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    tensor = torch.LongTensor(idxs)
    if use_gpu:
        return autograd.Variable(tensor).cuda()
    else:
        return autograd.Variable(tensor)


def load_pos(train=False, test=False, dev=False):
    if train == True:
        with open('./data/pos/train.txt', 'rb') as f:
            train_data = pickle.load(f)
        with open('./data/pos/train_labels.txt', 'rb') as f:
            train_labels = pickle.load(f)
        
        return train_data, train_labels

    if test == True:
        with open('./data/pos/test.txt', 'rb') as f:
            test_data = pickle.load(f)
        with open('./data/pos/test_labels.txt', 'rb') as f:
            test_labels = pickle.load(f)
        return test_data, test_labels
    if dev == True:
        with open('./data/pos/dev.txt', 'rb') as f:
            test_data = pickle.load(f)
        with open('./data/pos/dev_labels.txt', 'rb') as f:
            test_labels = pickle.load(f)
        return test_data, test_labels

    import pdb; pdb.set_trace()

def tag_indices(X, y):
    tag_to_idx = {START_TAG: 0, END_TAG: 1}
    idx_to_tag = {0: START_TAG, 1: END_TAG}
    for sent_tag in y:
        for tag in sent_tag:
            if tag not in tag_to_idx:
                idx_to_tag[len(tag_to_idx)] = tag
                tag_to_idx[tag] = len(tag_to_idx)

    with open('./models/pos_models/tag_to_idx', 'wb') as f:
        pickle.dump(tag_to_idx, f)
    return tag_to_idx, idx_to_tag

def char_dict(data):
    char_to_ix = {}
    for sent in data:
        for word in sent:
            word = word.lower()
            for character in word:
                if character not in char_to_ix:
                    char_to_ix[character] = len(char_to_ix)

    with open('./models/pos_models/char_to_ix.pkl', 'wb') as f:
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
    if len(chars2) == 0:
        print("Oh")
        return
    chars2_length = [len(c) for c in chars2]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
    for i, c in enumerate(chars2):
        chars2_mask[i, :chars2_length[i]] = c
    if use_gpu:
        chars2_mask = autograd.Variable(torch.cuda.LongTensor(chars2_mask))
    else:
        chars2_mask = autograd.Variable(torch.LongTensor(chars2_mask))
    return chars2_mask

def get_results(filename, model, data_X, data_Y, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu):
    correct = 0
    total = 0
    all_predicted = []
    all_targets= []

    len_test = len(data_X)
    print("Testing length", len_test)
    fname = filename+ str(epoch)+'.txt'
    with open(fname,'w') as f:
        for ind in range(len_test):
            sentence = data_X[ind]
            tags = data_Y[ind]

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix, tag=True)
            if use_gpu:
                caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence])).cuda()
            else:
                caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence]))
            if CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                prob, tag_seq = model(sentence_in, char_em, caps, 0)
                predicted = torch.LongTensor(tag_seq)
            else:
                tag_scores = model(sentence_in)
                prob, predicted = torch.max(tag_scores.data, 1)

            if use_gpu:
                all_predicted = all_predicted + (autograd.Variable(predicted).data.cpu().numpy().tolist())
                all_targets = all_targets + (targets.data.cpu().numpy().tolist())
            else:
                all_predicted.append(predicted)
                all_targets.append(targets)
            if use_gpu:
                correct += (predicted.cuda() == targets.data).sum()
            else:
                correct += (predicted == targets.data).sum()

            total += targets.size(0)
            for tag_ in range(len(sentence)):
                f.write(sentence[tag_] + " " + tags[tag_]+" "+idx_to_tag[predicted[tag_]]+"\n")
            f.write("\n")
    print("Accuracy is ", float(correct)/total)

def adjust_learning_rate(optimizer, epoch, LR):
    lr = LR / (1 + 0.5*epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():

    USE_CRF = False
    HIDDEN_DIM = 200
    CNN = True
    SENNA = False
    BIDIRECTIONAL = True
    bilstm_crf_cnn_flag = False

    training_data, y = load_pos(train=True)
    test_X, test_y = load_pos(test=True)
    dev_X, dev_y = load_pos(dev=True)

    char_to_ix = char_dict(training_data)
    tag_to_ix, idx_to_tag = tag_indices(training_data, y)

    if SENNA:
        EMBEDDING_DIM = 50
        emb_mat, word_to_ix = get_embeddings_matrix(training_data)
    else:
        print("Glove")
        EMBEDDING_DIM = 100
        emb_mat, word_to_ix = get_glove_matrix(training_data)

    if CNN == False and USE_CRF == False:
        print("Bilstm")
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))#, emb_mat, USE_CRF, BIDIRECTIONAL=True)
    if CNN == True and USE_CRF == False:
        print("Using BiLSTM-CNN-CRF")
        bilstm_crf_cnn_flag = True
        model = BILSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, tag_to_ix, CNN=True, BIDIRECTIONAL=True, use_gpu=use_gpu)
    if CNN == True and USE_CRF == True:
        bilstm_crf_cnn_flag = True
        print('Using BiLSTM-CRF-CNN')
        model = BiLSTM_CRF_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, tag_to_ix, USE_CRF, BIDIRECTIONAL, False, 0, CNN, use_gpu)

    if use_gpu:
        model = model.cuda()

    if not bilstm_crf_cnn_flag:
        loss_function = nn.NLLLoss()
    parameters = model.parameters()
    # parameters = filter(lambda p: model.requires_grad, model.parameters())
    # optimizer = optim.Adam(parameters, lr=0.001)
    learning_rate = 0.01
    optimizer = optim.SGD(parameters, lr=learning_rate)

    len_train = len(training_data)
    len_test = len(test_X)
    print("Number of train sentences ", len_train)
    print("Number of test sentences ", len_test)
    print('Training...')

    indices =[i for i in range(len_train)]
    shuffle(indices)
    last_time = time.time()
    fake_count = 0
    for epoch in range(500):
        loss_cal = 0.0
        count = 0
        for ind in (indices):
            count += 1
            sentence = training_data[ind]
            tags = y[ind]
            # data = all_data[ind]
            model.zero_grad()
            # THIS NOW HAPPENS IN FORWARD
            # model.hidden = model.init_hidden()#########
            sentence_in = prepare_sequence(sentence, word_to_ix)
            # sentence_in = data['sent']
            # targets = prepare_sequence(tags, tag_to_ix, tag=True)########
            # targets = data['tags']
            caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence]))
            targets = torch.LongTensor([tag_to_ix[t] for t in tags])

            # targets = data['tags']
            # import pdb; pdb.set_trace()
            if bilstm_crf_cnn_flag:
                char_in = prepare_words(sentence, char_to_ix)
                # char_in = data['chars']
                char_em = char_emb(char_in)
                # import pdb; pdb.set_trace()
                if use_gpu:
                    nll = model.neg_ll_loss(sentence_in.cuda(), targets, char_em.cuda(), caps.cuda(), 0.5)
                else:
                    nll = model.neg_ll_loss(sentence_in, targets, char_em, caps, 0.5)
                loss_cal += nll
                nll.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            elif CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                tag_scores = model(sentence_in,char_em)
            else:
                tag_scores = model(sentence_in)

            if not bilstm_crf_cnn_flag:
                loss = loss_function(tag_scores, targets)
                loss_cal += loss
                loss.backward()

            optimizer.step()
            if count % 1000 == 0 and ((count > 20000 and epoch==0) or (epoch!=0)):
                print(loss_cal)
                loss_cal = 0
                print(count, epoch)
                if SENNA:
                    # get_results('text/train_ner_bilstm_cnn', model, training_data, y, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
                    get_results('text/test_ner_bilstm_cnn', model, test_X, test_y, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
                else:
                    # get_results('pos_glove_text/train_ner_bilstm_cnn', model, training_data, y, ind, idx_to_tag, word_to_ix, tag_to_ix,char_to_ix, CNN, use_gpu)
                    # get_results('pos_glove_text/dev_ner_bilstm_cnn', model, dev_X, dev_y, ind, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
                    get_results('pos_glove_text/test_ner_bilstm_cnn', model, test_X, test_y, ind, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
            if count == 0 or count == 20000:
                fake_count += 1
                print("Annealing")
                adjust_learning_rate(optimizer, fake_count, learning_rate)

        print('Epoch {} took {:.3f}s'.format(epoch,time.time() - last_time))
        last_time = time.time()

if __name__ == '__main__':
    main()
