y_true = []
y_pred = []
pos_values = {}
from sklearn.metrics import classification_report

with open("test_pos_bilstm_cnn9000.txt", "r") as f:

    for line in f.readlines():
    	tokens = line.strip().split()
    	if len(tokens) != 0:
		    y_true += [tokens[2]]
		    y_pred += [tokens[1]]
val = classification_report(y_true, y_pred)

data = {}
count = 0

for sent in val.strip().split('\n'):
	tokens = sent.split()
	count += 1
	if count == 1 or count == len(val.strip().split('\n'))+1:
		continue
	if len(tokens) != 0:
		data[tokens[0]] = tokens[1]
	# except:
		# import pdb; pdb.set_trace()
del data['avg']
# del data['O']

import numpy as np
import matplotlib.pyplot as plt
import collections

fig = plt.subplots()
D = collections.Counter()

counter = 0
for key in sorted(data, key=data.__getitem__, reverse=True):
	D[key] = data[key]
	counter += 1
	if counter == 15:
		break

plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()))
plt.show()

