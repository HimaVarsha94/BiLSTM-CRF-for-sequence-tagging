from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle

"""
Parses train and dev sets into a dict where each of 7 keys maps
to a list of values for that key in order.

The keys are: user, countries, days, client, session, format, and time

Explained:
user: a B64 encoded, 8-digit, anonymized, unique identifier for each student (may include / or + characters)
countries: a pipe (|) delimited list of 2-character country codes from which this user has done exercises
days: the number of days since the student started learning this language on Duolingo
client: the student's device platform (one of: android, ios, or web)
session: the session type (one of: lesson, practice, or test; explanation below)
format: the exercise format (one of: reverse_translate, reverse_tap, or listen; see figures above)
time: the amount of time (in seconds) it took for the student to construct and
submit their whole answer (note: for some exercises, this can be null due to data logging issues)

Note: Train (293 unique users) and dev (292) sets have the exact same users except for
'aZwloMi1', who only appears in train
"""

DATA_PATH = '/Users/gosha/Desktop/data/duolingo_data/data_en_es/'

""" TRAIN """
feats = defaultdict(list)

with open(DATA_PATH + 'en_es.slam.20171218.train', 'r') as f:
    keys = []
    for line in f:
        tokens = line.split()
        if line[0] == '#':
            del tokens[0] # delete '#'
            if not feats:
                keys = [x.split(':')[0] for x in tokens]
            vals = [x.split(':')[1] for x in tokens]

            for k,v in zip(keys,vals):
                feats[k].append(v)


with open('../data/duolingo/train_seq_feats.pkl', 'wb') as f:
    pickle.dump(feats, f)


""" DEV """
feats = defaultdict(list)

## DEV labels are in separate file
with open(DATA_PATH + 'en_es.slam.20171218.dev', 'r') as f:
    keys = []
    for line in f:
        tokens = line.split()
        if line[0] == '#':
            del tokens[0] # delete '#'
            if not feats:
                keys = [x.split(':')[0] for x in tokens]
            vals = [x.split(':')[1] for x in tokens]

            for k,v in zip(keys,vals):
                feats[k].append(v)

with open('../data/duolingo/dev_seq_feats.pkl', 'wb') as f:
    pickle.dump(feats, f)
