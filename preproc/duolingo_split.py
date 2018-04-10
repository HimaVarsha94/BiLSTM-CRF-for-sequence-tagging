from sklearn.model_selection import train_test_split
import pickle

DATA_PATH = '/Users/gosha/Desktop/data/duolingo_data/data_en_es/'

""" TRAIN """
X = []
Y = []
user = 0

with open(DATA_PATH + 'en_es.slam.20171218.train', 'r') as f:
    sent = []
    y=[]
    for line in f:
        tokens = line.split()
        if line[0] == '#':
            if sent != []:
                X.append(sent)
                Y.append(y)
            sent = []
            y=[]
        elif len(tokens) == 7:
            sent.append(tokens[1])
            y.append(int(tokens[-1]))

with open(DATA_PATH + 'train_data.pkl', 'wb') as f:
    pickle.dump(X, f)

with open(DATA_PATH + 'train_data_labels.pkl', 'wb') as f:
    pickle.dump(Y, f)


""" DEV """
X = []
Y = []
user = 0

## DEV labels are in separate file
with open(DATA_PATH + 'en_es.slam.20171218.dev', 'r') as f_data:
    with open(DATA_PATH + 'en_es.slam.20171218.dev.key', 'r') as f_labels:
        sent = []
        y=[]
        for line in f_data:
            tokens = line.split()
            if line[0] == '#':
                if sent != []:
                    X.append(sent)
                    Y.append(y)
                sent = []
                y=[]
            elif len(tokens) == 6:
                label = int(f_labels.readline().split()[1])
                sent.append(tokens[1])
                y.append(label)

with open(DATA_PATH + 'dev_data.pkl', 'wb') as f:
    pickle.dump(X, f)

with open(DATA_PATH + 'dev_data_labels.pkl', 'wb') as f:
    pickle.dump(Y, f)

# print("num positive "+str(len([yy for yy in y for y in Y if yy==1])))
# print("num negative "+str(len([yy for yy in y for y in Y if yy==0])))
# print("sum "+str(len([yy for yy in y for y in Y])))

# import pdb; pdb.set_trace()
