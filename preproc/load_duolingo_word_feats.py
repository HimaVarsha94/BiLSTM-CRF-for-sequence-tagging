import pickle
from collections import Counter, defaultdict

## helper functions
def save_vocab(key,vals):
    vocab = {}
    for i,val in enumerate(vals):
        vocab[i+1] = val

    vocab[len(vocab)+1] = 'UNK'

    with open('../data/duolingo/vocabs_' + lang + '/'+key+'_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)


def replace_with_nums(data, vocab_list, pos_list, edge_label_list, edge_head_list, char_list):
    # max_word = 0
    for feats in data:
        # for i, w in enumerate(feats['words']):
        #     if w in vocab_list:
        #         ## no zeros allowed
        #         feats['words'][i] = vocab_list.index(w) + 1
        #     else:
        #         feats['words'][i] = len(vocab_list) + 1
            # if feats['words'][i] > max_word:
            #     max_word = feats['words'][i]

        for i, edge_label in enumerate(feats['edge_labels']):
            if edge_label in edge_label_list:
                feats['edge_labels'][i] = edge_label_list.index(edge_label) + 1
            else:
                feats['edge_labels'][i] = len(edge_label_list) + 1

        for i, pos in enumerate(feats['pos']):
            if pos in pos_list:
                feats['pos'][i] = pos_list.index(pos) + 1
            else:
                feats['pos'][i] = len(pos_list) + 1

        for i, word in enumerate(feats['chars']):
            for j, c in enumerate(word):
                if c in char_list:
                    feats['chars'][i][j] = char_list.index(c) + 1
                else:
                    feats['chars'][i][j] = len(char_list) + 1

        for i, e_h in enumerate(feats['edge_heads']):
            if e_h in edge_head_list:
                feats['edge_heads'][i] = edge_head_list.index(e_h) + 1
            else:
                feats['edge_heads'][i] = len(edge_head_list) + 1

    # print('max word',max_word)

    return data


def load_duolingo_word_feats(label):
    if label == 'train':
        with open('../data/duolingo/' + lang + '_train_allfeats_lowered.pkl', 'rb') as f:
            data = pickle.load(f)
        print("Train data length: "+str(len(data)))
        with open('../data/duolingo/' + lang + '_train_labels.pkl', 'rb') as f:
            data_labels = pickle.load(f)

    elif label == 'test' or label == 'dev':
        with open('../data/duolingo/' + lang + '_dev_allfeats_lowered.pkl', 'rb') as f:
            data = pickle.load(f)
        print("Dev data length: "+str(len(data)))
        with open('../data/duolingo/' + lang + '_dev_labels.pkl', 'rb') as f:
            data_labels = pickle.load(f)

    return data, data_labels


def process_duolingo_word_feats(train, test):
    vocab = Counter()
    pos_vocab = Counter()
    edge_label_vocab = Counter()
    char_vocab = Counter()
    edge_head_vocab = Counter()
    max_seq_len = 0
    max_char_len = 0
    max_edge_head = 0

    for feats in train:
        vocab.update(feats['words'])
        pos_vocab.update(feats['pos'])
        edge_label_vocab.update(feats['edge_labels'])
        edge_head_vocab.update(feats['edge_heads'])
        for w in feats['chars']:
            char_vocab.update(w)
            if len(w) > max_char_len:
                max_char_len = len(w)

        if len(feats['words']) > max_seq_len:
            max_seq_len = len(feats['words'])

        for e_h in feats['edge_heads']:
            if int(e_h) + 1 > max_edge_head:
                max_edge_head = int(e_h) + 1

    print('max seq len:', max_seq_len)
    print('max char len:', max_char_len)
    print('max edge head val:', max_edge_head)

    for w in list(vocab):
        if vocab[w] < 5:
            del (vocab[w])

    vocab_list = [tup[0] for tup in vocab.most_common()]
    with open('../data/duolingo/vocab_list.pkl', 'wb') as f:
        i2t = dict()
        for ind, token in enumerate(vocab_list):
            i2t[ind + 1] = token
        i2t[len(vocab_list) + 1] = "OOV"
        pickle.dump(i2t, f)

    pos_list = [tup[0] for tup in pos_vocab.most_common()]
    with open('../data/duolingo/pos_list.pkl', 'wb') as f:
        i2t = dict()
        for ind, token in enumerate(pos_list):
            i2t[ind + 1] = token
        i2t[len(pos_list) + 1] = "OOV"
        pickle.dump(i2t, f)

    timestamp_list = [tup[0] for tup in timestamp_vocab.most_common()]
    with open('../data/duolingo/timestamp_list.pkl', 'wb') as f:
        i2t = dict()
        for ind, token in enumerate(timestamp_list):
            i2t[ind + 1] = token
        i2t[len(timestamp_list) + 1] = "OOV"
        pickle.dump(i2t, f)

    edge_label_list = [tup[0] for tup in edge_label_vocab.most_common()]
    with open('../data/duolingo/edge_label_list.pkl', 'wb') as f:
        i2t = dict()
        for ind, token in enumerate(edge_label_list):
            i2t[ind + 1] = token
        i2t[len(edge_label_list) + 1] = "OOV"
        pickle.dump(i2t, f)

    char_list = [tup[0] for tup in char_vocab.most_common()]
    with open('../data/duolingo/char_list.pkl', 'wb') as f:
        i2t = dict()
        for ind, token in enumerate(char_list):
            i2t[ind + 1] = token
        i2t[len(char_list) + 1] = "OOV"
        pickle.dump(i2t, f)

    edge_head_list = [tup[0] for tup in edge_head_vocab.most_common()]
    with open('../data/duolingo/edge_head_list.pkl', 'wb') as f:
        i2t = dict()
        for ind, token in enumerate(edge_head_list):
            i2t[ind + 1] = token
        i2t[len(edge_head_list) + 1] = "OOV"
        pickle.dump(i2t, f)

    print('Word vocab length (incl unk)', len(vocab_list) + 2)
    print('POS vocab length', len(pos_list) + 2)
    print('Edge label vocab length', len(edge_label_list) + 2)
    print('Edge head vocab length', len(edge_head_list) + 2)
    print('Character vocab length', len(char_list) + 2)

    save_vocab('pos', pos_list)
    save_vocab('edge_label', edge_label_list)
    save_vocab('edge_head', edge_head_list)

    train = replace_with_nums(train, vocab_list, pos_list, edge_label_list, edge_head_list, char_list)
    test = replace_with_nums(test, vocab_list, pos_list, edge_label_list, edge_head_list, char_list)

    return train, test


if __name__ == '__main__':
    from sys import argv
    opts = {}
    while argv:
        if argv[0][0] == '-':
            opts[argv[0]] = argv[1]
        argv = argv[1:]

    lang = opts['--lang']

    train_data, train_labels = load_duolingo_word_feats('train')
    dev_data, dev_labels = load_duolingo_word_feats('dev')
    print(train_data[0])
    print(dev_data[0])

    train_data, dev_data = process_duolingo_word_feats(train_data, dev_data)

    print(train_data[0])
    print(dev_data[0])

    print(len(train_data))
    print(len(dev_data))

    with open('../data/duolingo/procd_' + lang + '_train_allfeats_lowered.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    with open('../data/duolingo/procd_' + lang + '_train_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f)

    with open('../data/duolingo/procd_' + lang + '_dev_allfeats_lowered.pkl', 'wb') as f:
        pickle.dump(dev_data, f)

    with open('../data/duolingo/procd_' + lang + '_dev_labels.pkl', 'wb') as f:
        pickle.dump(dev_labels, f)
