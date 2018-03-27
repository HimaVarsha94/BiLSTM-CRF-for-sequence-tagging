from sklearn.model_selection import train_test_split
import pickle

X = []
y = []
user = 0
with open('./data/duolingo/en_es.slam.20171218.train', 'r') as f:
	sent = []
	for line in f:
		tokens = line.split()
		if line[0] == '#':
			if sent != []:
				X.append(sent)
			sent = []
		elif len(tokens) == 7:
			sent.append(tokens[1])
			y.append(int(tokens[-1]))

with open('./data/duolingo/data.txt', 'wb') as f:
	pickle.dump(X, f)

with open('./data/duolingo/data_labels.txt', 'wb') as f:
	pickle.dump(y, f)
import pdb; pdb.set_trace()