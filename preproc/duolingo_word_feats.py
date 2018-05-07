from sklearn.model_selection import train_test_split
import pickle

from sys import argv
opts = {}
while argv:
    if argv[0][0] == '-':
        opts[argv[0]] = argv[1]
    argv = argv[1:]

lang = opts['--lang']

DATA_PATH = '../data/duolingo_orig/data_'+lang+'/'

""" TRAIN """
X = []
Y = []
user = 0

with open(DATA_PATH + lang + '.slam.20171218.train', 'r') as f:
    sent ={'words':[], 'pos':[], 'edge_labels':[], 'edge_heads':[], 'chars':[]}
    y=[]
    for line in f:
        tokens = line.split()
        if line[0] == '#':
            if sent != {'words':[], 'pos':[], 'edge_labels':[], 'edge_heads':[], 'chars':[]}:
                X.append(sent)
                Y.append(y)
            sent = {'words':[], 'pos':[], 'edge_labels':[], 'edge_heads':[], 'chars':[]}
            y=[]
        elif len(tokens) == 7:
            sent['words'].append(tokens[1])
            sent['pos'].append(tokens[2])
            sent['edge_labels'].append(tokens[4])
            sent['edge_heads'].append(tokens[5])
            sent['chars'].append(list(tokens[1]))
            # sent.append(tokens[1])
            y.append(int(tokens[-1]))

    X.append(sent)
    Y.append(y)

with open('../data/duolingo/'+lang+'_train_allfeats_lowered.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('../data/duolingo/' + lang + '_train_labels.pkl', 'wb') as f:
    pickle.dump(Y, f)


""" DEV """
X = []
Y = []
keys = []
user = 0

## DEV labels are in separate file
with open(DATA_PATH + lang + '.slam.20171218.dev', 'r') as f_data:
    with open(DATA_PATH + lang + '.slam.20171218.dev.key', 'r') as f_labels:
        sent = {'words':[], 'pos':[], 'edge_labels':[], 'edge_heads':[], 'chars':[]}
        y=[]
        for line in f_data:
            tokens = line.split()
            if line[0] == '#':
                if sent != {'words':[], 'pos':[], 'edge_labels':[], 'edge_heads':[], 'chars':[]}:
                    X.append(sent)
                    Y.append(y)
                sent = {'words':[], 'pos':[], 'edge_labels':[], 'edge_heads':[], 'chars':[]}
                y=[]
            elif len(tokens) == 6:
                label = int(f_labels.readline().split()[1])
                sent['words'].append(tokens[1])
                sent['pos'].append(tokens[2])
                sent['edge_labels'].append(tokens[4])
                sent['edge_heads'].append(tokens[5])
                sent['chars'].append(list(tokens[1]))
                # sent.append(tokens[1])
                y.append(label)
        X.append(sent)
        Y.append(y)

with open('../data/duolingo/' + lang + '_dev_allfeats_lowered.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('../data/duolingo/' + lang + '_dev_labels.pkl', 'wb') as f:
    pickle.dump(Y, f)

# print("num positive "+str(len([yy for yy in y for y in Y if yy==1])))
# print("num negative "+str(len([yy for yy in y for y in Y if yy==0])))
# print("sum "+str(len([yy for yy in y for y in Y])))

# import pdb; pdb.set_trace()
