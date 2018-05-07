import pickle
import math
from collections import Counter,defaultdict

time_splits = [1,2,3,4,5,10,15,20,25,30,math.inf]


## helper functions
def build_time_vocab(time_feats):
    for i,val in enumerate(time_feats):
        for j,split in enumerate(time_splits):
            if val == 'null':
                time_feats[i] = len(time_splits) + 1
                break
            elif float(val) // 60 < split:
                time_feats[i] = j + 1
                break
    return time_feats


def load_duolingo_seq_feats(label):
    if label == 'train':
        with open('../data/duolingo/es_en_train_seq_feats.pkl', 'rb') as f:
            seq_feats = pickle.load(f)
        print("Train data length: "+str(len(seq_feats['countries'])))

    elif label == 'test' or label == 'dev':
        with open('../data/duolingo/es_en_dev_seq_feats.pkl', 'rb') as f:
            seq_feats = pickle.load(f)
        print("Dev data length: "+str(len(seq_feats['countries'])))

    return seq_feats


def process_duolingo_seq_feats(train_feats, test_feats):
    exceptions = ['days', 'time']
    step_size = 100

    ## build vocab sets
    for key in train_feats.keys():
        unique_vals = list(set(train_feats[key]))
        if key in exceptions:
            if key == 'days':
                for i,val in enumerate(unique_vals):
                    unique_vals[i] = math.floor(float(val))
                unique_vals = list(set(unique_vals))
                # print(sorted(list(unique_vals)))

                for i,val in enumerate(train_feats[key]):
                    train_feats[key][i] = math.floor(float(val)) + 1

                for i,val in enumerate(test_feats[key]):
                    if math.floor(float(val)) in unique_vals:
                        test_feats[key][i] = math.floor(float(val)) + 1
                    else:
                        test_feats[key][i] = len(unique_vals) + 1

                print('{} vocab size: {}'.format(key,len(unique_vals)+2))

                vocab = {}
                for i,val in enumerate(unique_vals):
                    vocab[i+1] = val
                vocab[len(vocab)+1] = 'UNK'

                with open('../data/duolingo/vocabs_es_en/'+key+'_vocab.pkl', 'wb') as f:
                    pickle.dump(vocab, f)

            elif key == 'time':
                train_feats[key] = build_time_vocab(train_feats[key])
                test_feats[key] = build_time_vocab(test_feats[key])
                print('{} vocab size: {}'.format(key,len(list(set(train_feats[key])))+2))

                vocab = {1:'<1m', 2:'1-2m', 3:'2-3m', 4:'3-4m', 5:'4-5m', 6:'5-10m', 7:'10-15m',8:'15-20m', 9:'20-25m', 10:'25-30m', 11:'>30m'}
                with open('../data/duolingo/vocabs_es_en/'+key+'_vocab.pkl', 'wb') as f:
                    pickle.dump(vocab, f)

            continue

        print('{} vocab size: {}'.format(key,len(unique_vals)+2))

        for i,val in enumerate(train_feats[key]):
            train_feats[key][i] = unique_vals.index(val) + 1

        for i,val in enumerate(test_feats[key]):
            if val in unique_vals:
                test_feats[key][i] = unique_vals.index(val) + 1
            else:
                test_feats[key][i] = len(unique_vals) + 1

        vocab = {}
        for i,val in enumerate(unique_vals):
            vocab[i+1] = val

        if key in ['user', 'countries']:
            vocab[len(vocab)+1] = 'UNK'

        with open('../data/duolingo/vocabs_es_en/'+key+'_vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)

    return train_feats, test_feats


if __name__ == '__main__':
    train_feats = load_duolingo_seq_feats('train')
    dev_feats = load_duolingo_seq_feats('dev')
    for key in train_feats.keys():
        print(train_feats[key][0], end=' ')
    print()
    for key in dev_feats.keys():
        print(dev_feats[key][0], end=' ')
    print()

    train_feats, dev_feats = process_duolingo_seq_feats(train_feats, dev_feats)

    for key in train_feats.keys():
        print(train_feats[key][0], end=' ')
    print()
    for key in dev_feats.keys():
        print(dev_feats[key][0], end=' ')
    print()

    print(len(train_feats['countries']))
    print(len(dev_feats['countries']))

    ## change to list of dicts (from dict of vals)
    new_train_feats = [{'user':train_feats['user'][i], 'country':train_feats['countries'][i], 'days':train_feats['days'][i], 'client' :train_feats['client'][i], 'session':train_feats['session'][i], 'format':train_feats['format'][i], 'time':train_feats['time'][i]} for i in range(len(train_feats['countries']))]

    new_dev_feats = [{'user':dev_feats['user'][i], 'country':dev_feats['countries'][i], 'days':dev_feats['days'][i], 'client':dev_feats['client'][i], 'session':dev_feats['session'][i], 'format':dev_feats['format'][i], 'time':dev_feats['time'][i]} for i in range(len(dev_feats['countries']))]

    with open('../data/duolingo/procd_es_en_train_seq_feats.pkl', 'wb') as f:
        pickle.dump(new_train_feats, f)

    with open('../data/duolingo/procd_es_en_dev_seq_feats.pkl', 'wb') as f:
        pickle.dump(new_dev_feats, f)
