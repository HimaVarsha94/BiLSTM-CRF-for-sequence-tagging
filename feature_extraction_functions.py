import pickle

START_TAG = '<START>'
END_TAG = '<END>'

def extract_student_id():
    student_idx = {}
    with open('./data/duolingo/train_seq_feats.pkl', 'rb') as f:
        ids = pickle.load(f)

    student_ids = ids['user']
    for s_id in student_ids:
        if s_id not in student_idx:
            student_idx[s_id] = len(student_idx)
    return student_idx

def load_features():
    with open('./data/duolingo/train_seq_feats.pkl', 'rb') as f:
        ids = pickle.load(f)
    train_sids = ids['user']

    with open('./data/duolingo/dev_seq_feats.pkl', 'rb') as f:
        ids = pickle.load(f)
    test_sids = ids['user']
    return train_sids, test_sids

def cap_feature(s):
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def tag_indices(X, y, crf):
    if crf:
        tag_to_idx = {START_TAG: 0, END_TAG: 1}
        idx_to_tag = {0: START_TAG, 1: END_TAG}
    else:
        tag_to_idx = {}
        idx_to_tag = {}
    for sent_tag in y:
        for tag in sent_tag:
            if tag not in tag_to_idx:
                idx_to_tag[len(tag_to_idx)] = tag
                tag_to_idx[tag] = len(tag_to_idx)

    with open('./models/duolingo_models/tag_to_idx', 'wb') as f:
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

    with open('./models/duolingo_models/char_to_ix.pkl', 'wb') as f:
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
