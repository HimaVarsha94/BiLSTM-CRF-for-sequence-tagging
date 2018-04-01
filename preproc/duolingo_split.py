from sklearn.model_selection import train_test_split
import pickle

X = []
Y = []
user = 0

with open('../data/duolingo/en_es.slam.20171218.train', 'r') as f:
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

with open('../data/duolingo/data.txt', 'wb') as f:
    pickle.dump(X, f)

with open('../data/duolingo/data_labels.txt', 'wb') as f:
    pickle.dump(Y, f)

print("num positive "+str(len([yy for yy in y for y in Y if yy==1])))
print("num negative "+str(len([yy for yy in y for y in Y if yy==0])))
print("sum "+str(len([yy for yy in y for y in Y])))

# import pdb; pdb.set_trace()
