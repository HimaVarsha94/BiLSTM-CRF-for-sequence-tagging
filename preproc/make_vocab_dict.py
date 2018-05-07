import pickle

keys = ['client', 'countries', 'days', 'format', 'session', 'time', 'user', 'pos', 'edge_label', 'edge_head']
vocab_dict = {}

for key in keys:
    with open('../data/duolingo/vocabs_es_en/' + str(key) + '_vocab.pkl', 'rb') as f:
        vocab_dict[key] = pickle.load(f)

# print(vocab_dict.keys())
# print(vocab_dict['countries'])
# print(vocab_dict['user'][len(vocab_dict['user'])])
# print(vocab_dict['user'])
print(vocab_dict['pos'])

with open('../data/duolingo/vocabs_dict.pkl','wb') as f:
    pickle.dump(vocab_dict, f)
