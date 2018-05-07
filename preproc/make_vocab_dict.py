import pickle

vocab_dict = {}
for key in keys:
    with open('vocabs_es_en/' + str(key) + '_vocab.pkl', 'rb') as f:
        vocab_dict[key] = pickle.load(f)

# vocab_dict.keys()
# vocab_dict['countries']
# vocab_dict['user'][len(vocab_dict['user'])]
# vocab_dict['user']

with open('vocabs_dict.pkl','wb') as f:
    pickle.dump(vocab_dict, f)
