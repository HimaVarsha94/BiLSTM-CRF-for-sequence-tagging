import time, os
import numpy as np, pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.duolingo_model import BILSTM_CNN
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from random import shuffle
from datetime import timedelta
from feature_extraction_functions import *
#torch.cuda.set_device(3)
# torch.manual_seed(1)

START_TAG = '<START>'
END_TAG = '<END>'
use_gpu = 0


def get_embeddings_matrix(data):
    word_to_ix = {'unk':0}
    for sent in data:
        for word in sent:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    with open('./models/duolingo_models/word_to_ix.pkl', 'wb') as f:
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

    with open('./models/duolingo_models/duolingo_matrix', 'wb') as f:
        pickle.dump(embeddings_mat, f)

    return embeddings_mat, word_to_ix

def get_glove_matrix(data):
    word_to_ix = {'unk':0}
    for sent in data:
        for word in sent:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    with open('./models/duolingo_models/word_to_ix.pkl', 'wb') as f:
        pickle.dump(word_to_ix, f)
    #this contains a list of all senna obj embeddings
    with open('./embeddings/glove_obj.pkl', 'rb') as f:
        glove_obj = pickle.load(f)
    #embeddings matrix with each row corresponding to a word to pass to nn.embedding layer
    embeddings_mat = np.zeros((len(word_to_ix), 100))

    for word in word_to_ix:
        if word in glove_obj:
            embeddings_mat[word_to_ix[word]] = glove_obj[word]
        else:
            embeddings_mat[word_to_ix[word]] = np.random.rand(100)

    with open('./models/duolingo_models/duolingo_matrix', 'wb') as f:
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

def get_results(filename, model, data_X, data_Y, test_feats, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, sid_idx, CNN, USE_CRF, use_gpu):
    correct = 0
    total = 0
    all_predicted = []
    all_targets= []

    len_test = len(data_X)
    print("Testing length", len_test)
    fname = filename + str(epoch) + '.txt'
    with open(fname,'w') as f:
        for ind in range(len_test):
            sentence = data_X[ind]
            tags = data_Y[ind]

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix, tag=True)
            stud_id = test_feats[ind]
            student_ids = autograd.Variable(torch.LongTensor([sid_idx[stud_id]]*len(sentence)))

            if use_gpu:
                caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence])).cuda()
                student_ids = student_ids.cuda()
            else:
                caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence]))
            if CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                if USE_CRF:
                    prob, tag_seq = model(sentence_in, char_em, caps, student_ids, 0)
                    predicted = torch.LongTensor(tag_seq)
                else:
                    tag_scores = model(sentence_in, char_em, caps, student_ids, 0)
                    prob, predicted = torch.max(tag_scores.data, 1)

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
                f.write(sentence[tag_] + " " + str(tags[tag_])+" "+str(idx_to_tag[predicted[tag_]])+"\n")
            f.write("\n")

    ## Translate sequences into binary labels (0 correct 1 wrong)
    # gold_labels = torch.cat(all_targets).data - 2
    # pred_labels = torch.cat(all_predicted) - 2

    ## get actual binary labels for F1
    # with open('./data/duolingo/dev_binary_labels.pkl', 'rb') as f:
    #     gold_binary_labels = pickle.load(f)

    print("F1 score is ", compute_f1(targets.data, tag_scores.data[:,0]))
    print("Accuracy is ", float(correct)/total)
    print('AUROC: ', compute_auroc(targets.data, tag_scores.data[:,0]))


""" acc and f1 score code provided by duolingo, slightly modified since we don't have probabilities for each label """
def compute_f1(actual, predicted):
    """
    Computes the F1 score of your predictions. Note that we use 0.5 as the cutoff here.
    """
    num = len(actual)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for i in range(num):
        if actual[i] >= 0.5 and predicted[i] >= 0.5:
            true_positives += 1
        elif actual[i] < 0.5 and predicted[i] >= 0.5:
            false_positives += 1
        elif actual[i] >= 0.5 and predicted[i] < 0.5:
            false_negatives += 1
        else:
            true_negatives += 1

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        F1 = 2 * precision * recall / (precision + recall)
        print('Precision: ', precision)
        print('Recall: ', recall)
    except ZeroDivisionError:
        F1 = 0.0

    return F1


def compute_auroc(actual, predicted):
    """
    Computes the area under the receiver-operator characteristic curve.
    This code a rewriting of code by Ben Hamner, available here:
    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    # import pdb; pdb.set_trace()
    num = len(actual)
    temp = sorted([[predicted[i], actual[i]] for i in range(num)], reverse=True)

    sorted_predicted = [row[0] for row in temp]
    sorted_actual = [row[1] for row in temp]

    sorted_posterior = sorted(zip(sorted_predicted, range(len(sorted_predicted))))
    r = [0 for k in sorted_predicted]
    cur_val = sorted_posterior[0][0]
    last_rank = 0
    for i in range(len(sorted_posterior)):
        if cur_val != sorted_posterior[i][0]:
            cur_val = sorted_posterior[i][0]
            for j in range(last_rank, i):
                r[sorted_posterior[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_posterior)-1:
            for j in range(last_rank, i+1):
                r[sorted_posterior[j][1]] = float(last_rank+i+2)/2.0

    num_positive = len([0 for x in sorted_actual if x == 1])
    num_negative = num - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if sorted_actual[i] == 1])
    try:
        auroc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive))
    except ZeroDivisionError:
        auroc = 0.0

    return auroc


def load_duolingo(train=False, test=False):

    if train == True:
        with open('./data/duolingo/train_data.pkl', 'rb') as f:
            all_data = pickle.load(f)
        print("Train data length: "+str(len(all_data)))
        with open('./data/duolingo/train_data_labels.pkl', 'rb') as f:
            all_data_labels = pickle.load(f)
        return all_data, all_data_labels

    if test == True:
        with open('./data/duolingo/dev_data.pkl', 'rb') as f:
            all_data = pickle.load(f)
        print("Test data length: "+str(len(all_data)))
        with open('./data/duolingo/dev_data_labels.pkl', 'rb') as f:
            all_data_labels = pickle.load(f)
        return all_data, all_data_labels

    import pdb; pdb.set_trace()


def adjust_learning_rate(optimizer, epoch, LR):
    lr = LR / (1 + 0.5*epoch)
    print('New learning rate: {:.3f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():

    USE_CRF = False
    HIDDEN_DIM = 200
    CNN = True
    SENNA = False
    BIDIRECTIONAL = True
    bilstm_crf_cnn_flag = False

    training_data, y = load_duolingo(train=True)
    test_X, test_y = load_duolingo(test=True)
    sid_idx = extract_student_id()
    training_feats, test_feats = load_features()

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
        print('Using BiLSTM-CNN')
        model = BILSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, tag_to_ix, CNN=True, BIDIRECTIONAL=True, use_gpu=use_gpu, duolingo_student=True)
    if CNN == True and USE_CRF == True:
        print("Using BiLSTM-CNN-CRF")
        bilstm_crf_cnn_flag = True
        model = BILSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, tag_to_ix, CNN=True, BIDIRECTIONAL=True, use_gpu=use_gpu, duolingo_student=True)

    if use_gpu:
        model = model.cuda()

    if not bilstm_crf_cnn_flag:
        loss_function = nn.NLLLoss()
    parameters = model.parameters()
    # parameters = filter(lambda p: model.requires_grad, model.parameters())
    # optimizer = optim.Adam(parameters, lr=0.001)
    learning_rate = 0.1
    optimizer = optim.SGD(parameters, lr=learning_rate)

    len_train = len(training_data)
    len_test = len(test_X)
    print("Number of train sentences ", len_train)
    print("Number of test sentences ", len_test)
    print('Training...')


    last_time = time.time()
    lr_adjust_counter = -1
    for epoch in range(500):
        indices =[i for i in range(len_train)]
        shuffle(indices)
        loss_cal = 0.0
        count = -1
        for ind in indices:
            count += 1
            sentence = training_data[ind]

            tags = y[ind]
            model.zero_grad()
            # THIS NOW HAPPENS IN FORWARD
            sentence_in = prepare_sequence(sentence, word_to_ix)
            caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence]))
            targets = torch.LongTensor([tag_to_ix[t] for t in tags])
            stud_id = training_feats[ind]
            student_ids = autograd.Variable(torch.LongTensor([sid_idx[stud_id]]*len(sentence)))

            if bilstm_crf_cnn_flag:
                char_in = prepare_words(sentence, char_to_ix)
                # char_in = data['chars']
                char_em = char_emb(char_in)
                if use_gpu:
                    nll = model.neg_ll_loss(sentence_in.cuda(), targets, char_em.cuda(), caps.cuda(), student_ids.cuda(), 0.5)
                else:
                    nll = model.neg_ll_loss(sentence_in, targets, char_em, caps, student_ids, 0.5)
                loss_cal += nll
                nll.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            elif CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                tag_scores = model(sentence_in, char_em, caps, student_ids, 0.5)
            else:
                tag_scores = model(sentence_in)

            if not bilstm_crf_cnn_flag:
                loss = loss_function(torch.log(tag_scores), autograd.Variable(targets))
                loss_cal += loss
                loss.backward()

            optimizer.step()

            if count != 0 and count % (len(indices)//8) == 0:
                lr_adjust_counter += 1
                adjust_learning_rate(optimizer, lr_adjust_counter, learning_rate)

            # if count % 1000 == 0 and ((count > 20000 and epoch==0) or (epoch!=0)):
            if count != 0 and count % (len(indices)//8) == 0:
            # if count != 0 and count % 50000 == 0:
            # if epoch != 0 and count == 0:
                print('NLL Loss: {}'.format(float(loss_cal)))
                loss_cal = 0
                print('Epoch: {}, Sample: {}'.format(epoch, count))
                if SENNA:
                    # get_results('text/train_ner_bilstm_cnn', model, training_data, y, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
                    get_results('duolingo_text/test_duolingo_bilstm_cnn', model, test_X, test_y, test_feats, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, sid_idx, CNN, USE_CRF, use_gpu)
                else:
                    # get_results('pos_glove_text/train_ner_bilstm_cnn', model, training_data, y, ind, idx_to_tag, word_to_ix, tag_to_ix,char_to_ix, CNN, use_gpu)
                    # get_results('pos_glove_text/dev_ner_bilstm_cnn', model, dev_X, dev_y, ind, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
                    get_results('duolingo_glove_text/test_duolingo_bilstm_cnn', model, test_X, test_y, test_feats, ind, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, sid_idx, CNN, USE_CRF, use_gpu)
                print('Elapsed time in epoch: {}'.format(str(timedelta(seconds=int(time.time()-last_time)))))

        print('Epoch {} took {}'.format(epoch, str(timedelta(seconds=int(time.time()-last_time)))))
        get_results('duolingo_glove_text/train_duolingo_bilstm_cnn', model, training_data, y, ind, idx_to_tag, word_to_ix, tag_to_ix,char_to_ix, CNN, use_gpu)
        last_time = time.time()

if __name__ == '__main__':
    main()
