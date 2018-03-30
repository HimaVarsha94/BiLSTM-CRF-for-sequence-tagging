import time
import numpy as np, pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.lstm import LSTMTagger
from models.lstm_cnn import BILSTM_CNN
from models.bilstm_crf_cnn import BiLSTM_CRF_CNN
from sklearn.metrics import f1_score, precision_score, recall_score

torch.manual_seed(1)
use_gpu = 0

START_TAG = '<START>'
END_TAG = '<END>'

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags
def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True
def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    new_sentences = []
    for i, s in enumerate(sentences):
        # tags = [w[-1] for w in s]
        tags = s
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            new_sentences.append(new_tags)
            # import pdb; pdb.set_trace()
            # for word, new_tag in zip(s, new_tags):
                # word = new_tag
        else:
            raise Exception('Unknown tagging scheme!')
    return new_sentences

def get_embeddings_matrix(data):
    word_to_ix = {'unk':0}
    for sent in data:
        for word in sent:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    with open('./models/ner_models/word_to_ix.pkl', 'wb') as f:
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

    with open('./models/ner_models/ner_matrix', 'wb') as f:
        pickle.dump(embeddings_mat, f)

    return embeddings_mat, word_to_ix

def get_glove_matrix(data):
    word_to_ix = {'unk':0}
    for sent in data:
        for word in sent:
            word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    with open('./models/ner_models/word_to_ix.pkl', 'wb') as f:
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
            embeddings_mat[word_to_ix[word]] = np.zeros(100)

    with open('./models/ner_models/ner_matrix', 'wb') as f:
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

def ner_preprocess(datafile, senna=True):
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
            new_label.append(tokens[3])
    print(counter)
    return X, y

def load_ner(train=False, test=False):
    if train == True:
        train_data = open('./data/ner/eng.train')
        X_train, y_train = ner_preprocess(train_data)
        return X_train, y_train
    if test == True:
        print("testing data..")
        test_data = open('./data/ner/eng.testb')
        X_test, y_test = ner_preprocess(test_data)
        return X_test, y_test
    import pdb; pdb.set_trace()

def tag_indices(X, y):
    tag_to_idx = {START_TAG: 0, END_TAG: 1}
    idx_to_tag = {0: START_TAG, 1: END_TAG}
    for sent_tag in y:
        for tag in sent_tag:
            if tag not in tag_to_idx:
                idx_to_tag[len(tag_to_idx)] = tag
                tag_to_idx[tag] = len(tag_to_idx)

    with open('./models/ner_models/tag_to_idx', 'wb') as f:
        pickle.dump(tag_to_idx, f)
    return tag_to_idx, idx_to_tag

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
    if use_gpu:
        chars2_mask = autograd.Variable(torch.cuda.LongTensor(chars2_mask))
    else:
        chars2_mask = autograd.Variable(torch.LongTensor(chars2_mask))
    return chars2_mask

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

def get_results(filename, model, data_X, data_Y, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu):
    correct = 0
    total = 0
    all_predicted = []
    all_targets= []

    len_test = len(data_X)
    print("Testing length", len_test)
    with open(filename+ str(epoch)+'.txt','w') as f:
        for ind in range(len_test):
            sentence = data_X[ind]
            tags = data_Y[ind]

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix, tag=True)
            if CNN:
                char_in = prepare_words(sentence, char_to_ix)
                char_em = char_emb(char_in)
                prob, tag_seq = model(sentence_in,char_em)
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
            correct += (predicted == targets.data).sum()
            total += targets.size(0)
            for tag_ in range(len(sentence)):
                f.write(sentence[tag_] + " " + tags[tag_]+" "+idx_to_tag[predicted[tag_]]+"\n")
    print("F1 score weighted is ", f1_score(all_targets, all_predicted, average='weighted'))
    print("F1 score micro is ", f1_score(all_targets, all_predicted, average='micro'))
    print("Avg Precision ", precision_score(all_targets, all_predicted, average='weighted'))
    print("Avg Recall ", recall_score(all_targets, all_predicted, average='weighted'))

def main():

    USE_CRF = False
    HIDDEN_DIM = 600
    CNN = True
    SENNA = True
    BIDIRECTIONAL = False
    bilstm_crf_cnn_flag = False

    training_data, y = load_ner(train=True)
    test_X, test_y = load_ner(test=True)
    char_to_ix = char_dict(training_data)
    tag_to_ix, idx_to_tag = tag_indices(training_data, y)
    # y = update_tag_scheme(y, 'iobes')
    # test_y = update_tag_scheme(test_y, 'iobes')

    if SENNA:
        EMBEDDING_DIM = 50
        emb_mat, word_to_ix = get_embeddings_matrix(training_data)
    else:
        print("Glove")
        EMBEDDING_DIM = 100
        emb_mat, word_to_ix = get_glove_matrix(training_data)

    if CNN == False and USE_CRF == False:
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), emb_mat, USE_CRF)
    if CNN == True and USE_CRF == False:
        print("Using BiLSTM-CNN-CRF")
        bilstm_crf_cnn_flag = True
        model = BILSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix), emb_mat, tag_to_ix, CNN=True)
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
    optimizer = optim.SGD(parameters, lr=0.1)

    len_train = len(training_data)
    len_test = len(test_X)
    print("Number of train sentences ", len_train)
    print("Number of test sentences ", len_test)
    print('Training...')

    last_time = time.time()
    for epoch in range(500):
        loss_cal = 0.0
        # for ind in range(int(len_train)):
        #     sentence = training_data[ind]
        #     tags = y[ind]
        #     model.zero_grad()
        #     # THIS NOW HAPPENS IN FORWARD
        #     # model.hidden = model.init_hidden()
        #     sentence_in = prepare_sequence(sentence, word_to_ix)
        #     # targets = prepare_sequence(tags, tag_to_ix, tag=True)
        #     targets = torch.LongTensor([tag_to_ix[t] for t in tags])
        #     if bilstm_crf_cnn_flag:
        #         char_in = prepare_words(sentence, char_to_ix)
        #         char_em = char_emb(char_in)
        #         # import pdb; pdb.set_trace()
        #         nll = model.neg_ll_loss(sentence_in, targets, char_em)
        #         loss_cal += nll
        #         nll.backward()
        #     elif CNN:
        #         char_in = prepare_words(sentence, char_to_ix)
        #         char_em = char_emb(char_in)
        #         tag_scores = model(sentence_in,char_em)
        #     else:
        #         tag_scores = model(sentence_in)
        #
        #     if not bilstm_crf_cnn_flag:
        #         loss = loss_function(tag_scores, targets)
        #         loss_cal += loss
        #         loss.backward()
        #
        #     optimizer.step()

        print('Epoch {} took {:.3f}s'.format(epoch,time.time() - last_time))
        last_time = time.time()

        PATH = './models/ner_models/model_epoch' + str(epoch)
        torch.save(model.state_dict(), PATH)
        model.load_state_dict(torch.load(PATH))
        print(loss_cal)
        print("Finished one epoch and Testing!!")
        if SENNA:
            get_results('text/train_ner_bilstm_cnn', model, training_data, y, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)
            get_results('text/test_ner_bilstm_cnn', model, test_X, test_y, epoch, idx_to_tag, word_to_ix, tag_to_ix, CNN, use_gpu)
        else:
            get_results('glove_text/train_ner_bilstm_cnn', model, training_data, y, epoch, idx_to_tag, word_to_ix, tag_to_ix,char_to_ix, CNN, use_gpu)
            get_results('glove_text/test_ner_bilstm_cnn', model, test_X, test_y, epoch, idx_to_tag, word_to_ix, tag_to_ix, char_to_ix, CNN, use_gpu)

        print('Testing took {:.3f}s'.format(time.time() - last_time))
        last_time = time.time()

if __name__ == '__main__':
    main()
