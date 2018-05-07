import time, datetime, pickle
import numpy as np
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
from sys import argv
import pdb

START_TAG = '<START>'
END_TAG = '<END>'


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
    #this contains a list of all glove obj embeddings
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
        if tag is False:
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


def get_results(filename, model, test_data_feats, test_seq_feats, test_labels, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, sid_idx, CNN, USE_CRF, use_gpu):
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []

    with open('./data/duolingo/vocabs_dict.pkl', 'rb') as f:
        vocabs = pickle.load(f)

    # test_data_feats, test_labels, test_feats
    len_test = len(test_data_feats)
    print("Testing length", len_test)
    fname = filename+'_'+str(learning_rate)+'_'+str(epoch)+'_'+datetime.datetime.now().isoformat()[:19]+'.txt'
    with open(fname,'w') as f:
        f.write('word correct predicted probability user country days client session format time pos edge_label edge_head\n')
        for ind in range(len_test):
            sentence = test_data_feats[ind]['words']
            tags = test_labels[ind]

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix, tag=True)

            feats = {}
            ## seq feats
            for key in test_seq_feats[ind].keys():
                if use_gpu:
                    feats[key] = autograd.Variable(torch.LongTensor([test_seq_feats[ind][key]]*len(sentence))).cuda()
                else:
                    feats[key] = autograd.Variable(torch.LongTensor([test_seq_feats[ind][key]]*len(sentence)))

            ## token feats
            for key in ['pos','edge_labels','edge_heads']:
                if use_gpu:
                    feats[key] = autograd.Variable(torch.LongTensor(test_data_feats[ind][key])).cuda()
                else:
                    feats[key] = autograd.Variable(torch.LongTensor(test_data_feats[ind][key]))

            if use_gpu:
                caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence])).cuda()
            else:
                caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence]))

            if CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                if USE_CRF:
                    prob, tag_seq = model(sentence_in, char_em, caps, feats, 0)
                    predicted = torch.LongTensor(tag_seq)
                    tag_scores = prob
                else:
                    tag_scores = model(sentence_in, char_em, caps, feats, 0)
                    prob, predicted = torch.max(tag_scores.data, 1)
            else:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                tag_scores = model(sentence_in, char_em, caps, feats, 0)
                prob, predicted = torch.max(tag_scores.data, 1)

            if use_gpu:
                if USE_CRF:
                    all_predicted += tag_scores.data.cpu().numpy().tolist()
                else:
                    all_predicted += tag_scores.data.cpu().numpy()[:,1].tolist()
                all_targets += targets.data.cpu().numpy().tolist()
            else:
                if USE_CRF:
                    all_predicted += tag_scores.data.numpy().tolist()
                else:
                    all_predicted += tag_scores.data.numpy()[:,1].tolist()
                all_targets += targets.data.cpu().numpy().tolist()

            if use_gpu:
                correct += (predicted.cuda() == targets.data).sum()
            else:
                correct += (predicted == targets.data).sum()

            predicted = predicted.cpu().tolist()
            # pdb.set_trace()

            total += targets.size(0)

            string = "{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n"
            for tag_ in range(len(sentence)):
                format_args = [sentence[tag_], tags[tag_], idx_to_tag[predicted[tag_]],
                               prob[tag_],
                               vocabs['user'][test_seq_feats[ind]['user']],
                               vocabs['countries'][test_seq_feats[ind]['country']],
                               vocabs['days'][test_seq_feats[ind]['days']],
                               vocabs['client'][test_seq_feats[ind]['client']],
                               vocabs['session'][test_seq_feats[ind]['session']],
                               vocabs['format'][test_seq_feats[ind]['format']],
                               vocabs['time'][test_seq_feats[ind]['time']],
                               vocabs['pos'][test_data_feats[ind]['pos'][tag_]],
                               vocabs['edge_label'][test_data_feats[ind]['edge_labels'][tag_]],
                               vocabs['edge_head'][test_data_feats[ind]['edge_heads'][tag_]]]
                f.write(string.format(*format_args))
            f.write("\n")

            del sentence, tags, sentence_in, feats, caps,
            char_in, char_em, prob, predicted
            try:
                del tag_scores
            except:
                pass

    print('Len all_predicted: {}'.format(len(all_predicted)))
    print("F1 score is ", compute_f1(all_targets, all_predicted))
    print("Accuracy is ", float(correct)/total)
    print('AUROC: ', compute_auroc(all_targets, all_predicted))


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


def load_duolingo(label):
    if label == 'train':
        with open('./data/duolingo/procd_es_en_train_allfeats_lowered.pkl', 'rb') as f:
            train_data = pickle.load(f)

        with open('./data/duolingo/procd_es_en_train_labels.pkl', 'rb') as f:
            train_labels = pickle.load(f)

        with open('./data/duolingo/procd_es_en_train_seq_feats.pkl', 'rb') as f:
            train_feats = pickle.load(f)

        return train_data, train_feats, train_labels

    elif label == 'test' or label == 'dev':
        with open('./data/duolingo/procd_es_en_dev_allfeats_lowered.pkl', 'rb') as f:
            test_data = pickle.load(f)

        with open('./data/duolingo/procd_es_en_dev_labels.pkl', 'rb') as f:
            test_labels = pickle.load(f)

        with open('./data/duolingo/procd_es_en_dev_seq_feats.pkl', 'rb') as f:
            test_feats = pickle.load(f)

        return test_data, test_feats, test_labels

    elif label == 'vocabs':
        with open('./data/duolingo/vocabs_es_en/client_vocab.pkl', 'rb') as f:
            client_vocab = pickle.load(f)
        with open('./data/duolingo/vocabs_es_en/countries_vocab.pkl', 'rb') as f:
            countries_vocab = pickle.load(f)
        with open('./data/duolingo/vocabs_es_en/days_vocab.pkl', 'rb') as f:
            days_vocab = pickle.load(f)
        with open('./data/duolingo/vocabs_es_en/format_vocab.pkl', 'rb') as f:
            format_vocab = pickle.load(f)
        with open('./data/duolingo/vocabs_es_en/session_vocab.pkl', 'rb') as f:
            session_vocab = pickle.load(f)
        with open('./data/duolingo/vocabs_es_en/time_vocab.pkl', 'rb') as f:
            time_vocab = pickle.load(f)
        with open('./data/duolingo/vocabs_es_en/user_vocab.pkl', 'rb') as f:
            user_vocab = pickle.load(f)

        return {'client':client_vocab, 'countries':countries_vocab, 'days':days_vocab,
                'format':format_vocab, 'session':session_vocab, 'time':time_vocab, 'user':user_vocab}


def adjust_learning_rate(optimizer, epoch, LR):
    lr = LR / (1 + 0.5*epoch)
    print('New learning rate: {:.3f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    USE_CRF = False
    HIDDEN_DIM = 200
    CNN = False
    SENNA = False
    DROPOUT = 0.1
    BIDIRECTIONAL = True
    bilstm_crf_cnn_flag = False

    train_data_feats, train_seq_feats, train_labels = load_duolingo('train')
    test_data_feats, test_seq_feats, test_labels = load_duolingo('dev')

    sid_idx = extract_student_id()
    training_feats, test_feats = load_features()

    train_data = [train_data_feats[i]['words'] for i in range(len(train_data_feats))]
    test_data = [test_data_feats[i]['words'] for i in range(len(test_data_feats))]

    char_to_ix = char_dict(train_data)
    tag_to_ix, idx_to_tag = tag_indices(train_data, train_labels, USE_CRF)

    vocab_sizes = {'user': 2645,
                   'country': 201,
                   'days': 30,
                   'client': 4,
                   'session': 4,
                   'format': 4,
                   'time': 14,
                   'pos': 17,
                   'edge_labels': 33,
                   'edge_heads': 16}

    if SENNA:
        EMBEDDING_DIM = 50
        emb_mat, word_to_ix = get_embeddings_matrix(train_data)
    else:
        print("Glove")
        EMBEDDING_DIM = 100
        emb_mat, word_to_ix = get_glove_matrix(train_data)

    if CNN == False and USE_CRF == False:
        print("Bilstm")
        # model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))#, emb_mat, USE_CRF, BIDIRECTIONAL=True)
        model = BILSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, tag_to_ix, vocab_sizes, USE_CRF=USE_CRF, CNN=CNN, BIDIRECTIONAL=BIDIRECTIONAL, use_gpu=use_gpu, duolingo_student=True)
    if CNN == True and USE_CRF == False:
        print('Using BiLSTM-CNN')
        model = BILSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, tag_to_ix, vocab_sizes, USE_CRF=USE_CRF, CNN=CNN, BIDIRECTIONAL=BIDIRECTIONAL, use_gpu=use_gpu, duolingo_student=True)
    if CNN == True and USE_CRF == True:
        print("Using BiLSTM-CNN-CRF")
        bilstm_crf_cnn_flag = True
        model = BILSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, tag_to_ix, vocab_sizes, USE_CRF=USE_CRF, CNN=CNN, BIDIRECTIONAL=BIDIRECTIONAL, use_gpu=use_gpu, duolingo_student=True)

    if use_gpu:
        model = model.cuda()

    if not bilstm_crf_cnn_flag:
        loss_function = nn.NLLLoss()
    parameters = model.parameters()
    #learning_rate = 0.05
    optimizer = optim.SGD(parameters, lr=learning_rate)

    len_train = len(train_data)
    len_test = len(test_data)
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
            sentence = train_data[ind]

            tags = train_labels[ind]
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            caps = autograd.Variable(torch.LongTensor([cap_feature(w) for w in sentence]))
            targets = torch.LongTensor([tag_to_ix[t] for t in tags])

            feats = {}
            ## seq feats
            for key in train_seq_feats[ind].keys():
                if use_gpu:
                    feats[key] = autograd.Variable(torch.LongTensor([train_seq_feats[ind][key]]*len(sentence))).cuda()
                else:
                    feats[key] = autograd.Variable(torch.LongTensor([train_seq_feats[ind][key]]*len(sentence)))

            ## token feats
            for key in ['pos','edge_labels','edge_heads']:
                if use_gpu:
                    feats[key] = autograd.Variable(torch.LongTensor(train_data_feats[ind][key])).cuda()
                else:
                    feats[key] = autograd.Variable(torch.LongTensor(train_data_feats[ind][key]))

            if bilstm_crf_cnn_flag:
                char_in = prepare_words(sentence, char_to_ix)
                # char_in = data['chars']
                char_em = char_emb(char_in)
                if use_gpu:
                    nll = model.neg_ll_loss(sentence_in.cuda(), targets, char_em.cuda(), caps.cuda(), feats, DROPOUT)
                else:
                    nll = model.neg_ll_loss(sentence_in, targets, char_em, caps, feats, DROPOUT)

                loss_cal += float(nll.cpu().detach().numpy())
                nll.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            elif CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                if use_gpu:
                    tag_scores = model(sentence_in.cuda(), char_em.cuda(),
                            caps.cuda(), feats, DROPOUT)
                else:
                    tag_scores = model(sentence_in, char_em, caps, feats, DROPOUT)
            else:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                if use_gpu:
                    tag_scores = model(sentence_in.cuda(), char_em.cuda(),
                            caps.cuda(), feats, DROPOUT)
                else:
                    tag_scores = model(sentence_in, char_em, caps, feats, DROPOUT)

            if not bilstm_crf_cnn_flag:
                if use_gpu:
                    loss = loss_function(torch.log(tag_scores),
                        autograd.Variable(targets).cuda())
                else:
                    loss = loss_function(torch.log(tag_scores),
                        autograd.Variable(targets))
                loss_cal += float(loss.cpu().detach().numpy())
                loss.backward()

            optimizer.step()

            if epoch > 5 and count == 0:
                lr_adjust_counter += 1
                adjust_learning_rate(optimizer, lr_adjust_counter, learning_rate)

            del sentence, char_in, char_em, targets, caps, sentence_in, tags, feats

            try:
                del tag_scores, loss
            except:
                pass

            try:
                del nll
            except:
                pass

            # if count % 1000 == 0 and ((count > 20000 and epoch==0) or (epoch!=0)):
            # if count != 0 and count % ((len(indices)-1)//2) == 0:
            if count != 0 and count % 1 == 0:
            # if epoch != 0 and count == 0:
                print('NLL Loss: {}'.format(float(loss_cal)))
                loss_cal = 0.
                print('Epoch: {}, Sample: {}'.format(epoch, count))
                if SENNA:
                    # get_results('text/train_ner_bilstm_cnn', model, train_data, y, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
                    get_results('duolingo_text/test_duolingo_bilstm_cnn', model, test_data_feats, test_seq_feats, test_labels, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, sid_idx, CNN, USE_CRF, use_gpu)
                else:
                    # get_results('pos_glove_text/train_ner_bilstm_cnn', model, train_data, y, ind, idx_to_tag, word_to_ix, tag_to_ix,char_to_ix, CNN, use_gpu)
                    # get_results('pos_glove_text/dev_ner_bilstm_cnn', model, dev_X, dev_y, ind, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
                    get_results('duolingo_glove_text/test_duolingo_bilstm_cnn', model, test_data_feats, test_seq_feats, test_labels, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, sid_idx, CNN, USE_CRF, use_gpu)
                print('Elapsed time in epoch: {}'.format(str(timedelta(seconds=int(time.time()-last_time)))))

        print('Epoch {} took {}'.format(epoch, str(timedelta(seconds=int(time.time()-last_time)))))
        #get_results('duolingo_glove_text/test_duolingo_bilstm_cnn', model, train_data, train_labels, training_feats, ind, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, sid_idx, CNN, USE_CRF, use_gpu)
        #get_results('duolingo_glove_text/test_duolingo_bilstm_cnn', model, test_data_feats, test_seq_feats, test_labels, ind, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, sid_idx, CNN, USE_CRF, use_gpu)
        last_time = time.time()


if __name__ == '__main__':
    opts = {}
    while argv:
        if argv[0][0] == '-':
            opts[argv[0]] = argv[1]
        argv = argv[1:]

    try:
        use_gpu = int(opts['--gpu'])
        if use_gpu and '--device' in opts:
            torch.cuda.set_device(int(opts['--device']))

        learning_rate = float(opts['--lr'])

    except(KeyError):
        print('Usage:')
        print('python duolingo_train.py --gpu 0 --lr 0.05            # for cpu')
        print('python duolingo_train.py --gpu 1 --device 2 --lr 0.05 # for gpu, --device can be omitted')
        exit()

    main()
