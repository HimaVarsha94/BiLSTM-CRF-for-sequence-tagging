import pickle
import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder

from Dataset import Dataset
from cluf_pymodel_batch_user import CLUF
from evaluation import evaluate_metrics
from evaluation import compute_acc, compute_avg_log_loss, compute_auroc, compute_f1, compute_auc

import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.optim as optim
from constant import PAD

'''
To run, command: python train_batch_user.py 

'''
print(PAD)


def load_data():
    with open('data_es_en/procd_es_en_train_allfeats_lowered.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('data_es_en/procd_es_en_train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    with open('data_es_en/procd_es_en_dev_allfeats_lowered.pkl', 'rb') as f:
        dev_data = pickle.load(f)

    with open('data_es_en/procd_es_en_dev_labels.pkl', 'rb') as f:
        dev_labels = pickle.load(f)

    return train_data, train_labels, dev_data, dev_labels


def load_ordered_data():
    with open('./data_es_en/ordered_feats/ordered_es_en_train_allfeats_lowered.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('./data_es_en/ordered_feats/ordered_es_en_dev_allfeats_lowered.pkl', 'rb') as f:
        dev_data = pickle.load(f)

    with open('./data_es_en/ordered_feats/ordered_es_en_train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    with open('./data_es_en/ordered_feats/ordered_es_en_dev_labels.pkl', 'rb') as f:
        dev_labels = pickle.load(f)

    with open('./data_es_en/ordered_feats/ordered_es_en_train_seq_feats.pkl', 'rb') as f:
        train_feats = pickle.load(f)
        # print(train_feats[:10])

    with open('./data_es_en/ordered_feats/ordered_es_en_dev_seq_feats.pkl', 'rb') as f:
        dev_feats = pickle.load(f)
        # print(dev_feats[:10])

    return train_data, train_labels, dev_data, dev_labels, train_feats, dev_feats


# return train_data[:500], train_labels[:500], train_data[:500], train_labels[:500]

def preprocess(data, max_size):
    while (True):
        if len(data) < max_size:
            data.append(0)
        else:
            break
    return reshaping(data)


def transform_targets(data):
    res = np.zeros((len(data), 2))
    for i, sample in enumerate(data):
        if sample == 1:
            res[i, 1] = 1
        else:
            res[i, 0] = 1
    return res


def make_variable(data, use_gpu):
    if use_gpu:
        return autograd.Variable(torch.LongTensor(data)).cuda()
    else:
        return autograd.Variable(torch.LongTensor(data))


# def load_data():
#    train_data = [{'words': [0, 1, 2, 0, 3], 'pos': [0, 1, 2, 0, 3], 'edge_labels': [0, 1, 2, 0, 3]},
#                  {'words': [4, 5, 6, 7], 'pos': [4, 5, 6, 7], 'edge_labels': [4, 5, 6, 7]}]
#    train_labels = [[1, 0, 0, 1, 0], [0, 0, 1, 0]]
#
#    dev_data = train_data
#    dev_labels = train_labels
#
#    return train_data, train_labels, dev_data, dev_labels


# def load_data():
#     train_data = [{'words': [19], 'pos': [5], 'edge_labels': [4], 'edge_heads': [3], 'chars': [[25, 11, 20]]},
#                   {'words': [75], 'pos': [5], 'edge_labels': [4], 'edge_heads': [3], 'chars': [[10, 23, 9, 1]]},
#                   {'words': [139], 'pos': [5], 'edge_labels': [4], 'edge_heads': [3],
#                    'chars': [[10, 11, 24, 6, 12, 1, 4]]},
#                   {'words': [72, 56], 'pos': [5, 2], 'edge_labels': [4, 2], 'edge_heads': [4, 4],
#                    'chars': [[10, 11, 24, 5, 2, 4], [3, 6, 8, 9, 3, 5, 2, 4]]},
#                   {'words': [2, 243, 1, 318], 'pos': [2, 1, 3, 8], 'edge_labels': [2, 3, 5, 1],
#                    'edge_heads': [3, 5, 5, 1],
#                    'chars': [[2, 5], [10, 11, 10, 18, 8, 5, 5, 1], [2, 4], [6, 2, 17, 7, 1]]},
#                   {'words': [4, 25, 314], 'pos': [4, 3, 1], 'edge_labels': [3, 1, 6], 'edge_heads': [3, 1, 3],
#                    'chars': [[16, 1], [12, 2, 6, 17, 1], [13, 5, 3, 12, 1, 4]]},
#                   {'words': [139, 316], 'pos': [5, 4], 'edge_labels': [4, 1], 'edge_heads': [3, 1],
#                    'chars': [[10, 11, 24, 6, 12, 1, 4], [19, 3, 4, 1, 4]]},
#                   {'words': [16, 184], 'pos': [2, 1], 'edge_labels': [2, 1], 'edge_heads': [3, 1],
#                    'chars': [[5, 3, 4], [19, 2, 6, 12, 3, 6, 3, 4]]},
#                   {'words': [1, 7, 229], 'pos': [3, 2, 1], 'edge_labels': [5, 2, 1], 'edge_heads': [4, 4, 1],
#                    'chars': [[2, 4], [9, 8], [2, 4, 10, 7, 8, 12, 1, 7, 8, 1]]},
#                   {'words': [2, 167], 'pos': [2, 1], 'edge_labels': [2, 1], 'edge_heads': [3, 1],
#                    'chars': [[2, 5], [2, 4, 13, 2, 21, 1]]}]
#
#     train_labels = [[0], [0], [0], [0, 0], [1, 1, 0, 0], [0, 0, 0], [0, 0], [0, 0], [0, 0, 0], [0, 0]]
#
#     dev_data = train_data
#     dev_labels = train_labels
#
#     return train_data, train_labels, dev_data, dev_labels


def test(dev_data_batch, cluf_model, use_user=True):
    cluf_model.eval()
    all_predicted = []
    all_targets = []

    for i in range(len(dev_data_batch)):
        batch = dev_data_batch[i]

        words_length_tuple = batch[0]  # (words, lengths)
        pos = batch[1]
        edges = batch[2]
        targets = batch[3]

        if use_user:
            users = batch[5]
            countries = batch[6]
            days = batch[7]

            # assert (users != None and countries != None and days != None)
        else:
            users, countries, days = None, None, None

        _, prob = cluf_model(words_length_tuple, pos, edges, user_info=(users, countries, days))

        this_predicted, this_target = manipulate_valid_pred(prob, targets)

        # prob=prob.contiguous().view(-1, 1)
        # weights = targets.ne(-1)

        all_predicted = all_predicted + this_predicted
        all_targets = all_targets + this_target
        if i % 1000 == 0:
            print("testing finished " + str(i))

    # all_targets = [item for sublist in dev_labels for item in sublist]
    results = evaluate_metrics(all_targets, all_predicted)
    for key in results:
        print(str(key) + ": " + str(results[key]))
    return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--num_epochs', dest='num_epochs', type=str, default=50,
                        help='Number of epochs to train')
    parser.add_argument("--gpus", dest='gpus', default=0, type=int,
                        help="Use CUDA")
    return parser.parse_args()


def masked_NLLLoss(log_softmax_tag_scores, targets):
    weights = targets.ne(-1).float()
    targets = targets.eq(1).long()

    class_count = log_softmax_tag_scores.shape[-1]
    # print("log_softmax_tag_scores shape " + str(log_softmax_tag_scores))
    # print("contiguous " + str(log_softmax_tag_scores.contiguous().view(-1, class_count)))

    cont_log = log_softmax_tag_scores.contiguous().view(-1, class_count)

    # print("targets shape " + str(targets))
    # print("contiguous " + str(targets.contiguous().view(-1, 1)))

    cont_targets = targets.contiguous().view(-1, 1)

    # print("weights " + str(weights))
    # print("contiguous " + str(weights.contiguous().view(-1, 1)))
    cont_weights = weights.contiguous().view(-1, 1)

    # print("targets.unsqueeze(1))" + str(targets.unsqueeze(2)))
    # print(" -log_softmax_tag_scores.gather(1, targets.unsqueeze(1))" + str(
    #     -torch.gather(cont_log, 1, index=cont_targets)))
    # input()

    losses = -torch.gather(cont_log, 1, index=cont_targets)  # negative log likelihood
    losses = losses * cont_weights
    # print(losses)
    # print(losses.sum())
    # input()
    return losses.sum()


def manipulate_valid_pred(prob, targets):
    # print("manipulate_valid_pred")
    # print("initial prob "+str(prob))
    # print("initial targets "+str(targets))
    prob = prob.data[:, :, 1].contiguous().view(-1)
    # print("selected class 1 prob "+str(prob))

    targets = targets.data.contiguous().view(-1)
    # print("flattened targets "+str(targets))
    targets = targets.cpu().numpy()
    # print(targets)
    # print(np.where(targets!=-1))
    prob = prob.cpu().numpy()
    valid_index = np.where(targets != -1)[0]
    # print(prob)
    return prob[valid_index].tolist(), targets[valid_index].tolist()


def main():
    args = parse_arguments()

    use_user = True

    if use_user:
        train_data, train_labels, dev_data, dev_labels, train_feats, dev_feats = load_ordered_data()

    gpu_flag = int(args.gpus)
    cluf_model = CLUF(gpu_flag, use_user)
    if gpu_flag:
        torch.cuda.set_device(0)
        cluf_model = cluf_model.cuda()

    loss_function = nn.NLLLoss()

    parameters = cluf_model.parameters()
    learning_rate = 0.001
    batch_size = 32

    optimizer = optim.Adam(parameters, lr=learning_rate)

    train_data_batch = Dataset(train_data, train_labels, gpu_flag, batch_size, eval=False, session_feats=train_feats)
    dev_data_batch = Dataset(dev_data, dev_labels, gpu_flag, batch_size, eval=True, session_feats=dev_feats)

    # import pdb; pdb.set_trace()
    for epoch_ in range(args.num_epochs):
        print("Epoch: {}".format(epoch_))
        train_data_batch.shuffle()
        # total_loss = 0

        for i in range(len(train_data_batch)):
            batch = train_data_batch[i]

            words_length_tuple = batch[0]  # (words, lengths)
            pos = batch[1]
            edges = batch[2]
            targets = batch[3]

            if use_user:
                users = batch[5]
                countries = batch[6]
                days = batch[7]

                # print("within this batch " + str(users))
                # print("within this batch " + str(countries))
                # print("within this batch " + str(days))

                # assert (users != None and countries != None and days != None)
            else:
                users, countries, days = None, None, None

            # print("targets loaded " + str(targets))
            cluf_model.train()
            cluf_model.zero_grad()

            log_softmax_tag_scores, _ = cluf_model(words_length_tuple, pos, edges, user_info=(users, countries, days))
            loss = masked_NLLLoss(log_softmax_tag_scores, targets)

            # manipulate_valid_pred(prob, targets)

            # total_loss += loss
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print("training finished " + str(i) + " batches (batch_size 32)")
            if i % 10000 == 0:
                print("testing, after " + str(i) + " batches")
                test(dev_data_batch, cluf_model, use_user)
        test(dev_data_batch, cluf_model, use_user)
            # print(total_loss)
    return


def reshaping(data):
    data = np.array(data)
    return data.reshape(1, data.shape[0])


if __name__ == '__main__':
    main()
