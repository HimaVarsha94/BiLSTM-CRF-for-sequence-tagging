import pickle
import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder
import os
import re
from Dataset import Dataset
from cluf_pymodel_batch_user_format import CLUF
from evaluation import evaluate_metrics
from evaluation import compute_acc, compute_avg_log_loss, compute_auroc, compute_f1, compute_auc
import sys
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from constant import PAD
import time as time_lib

print(PAD)
F1 = 0.0
AUROC = 0.0


class selfDict(dict):
    __missing__ = lambda self, key: key


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


def load_ordered_data(directory='./data_es_en/ordered_feats/'):
    if 'en_es' in directory:
        name = 'en_es'
    elif 'es_en' in directory:
        name = 'es_en'
    elif 'fr_en' in directory:
        name = 'fr_en'

    with open(directory + 'ordered_' + name + '_train_allfeats_lowered.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open(directory + 'ordered_' + name + '_dev_allfeats_lowered.pkl', 'rb') as f:
        dev_data = pickle.load(f)

    with open(directory + 'ordered_' + name + '_train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    with open(directory + 'ordered_' + name + '_dev_labels.pkl', 'rb') as f:
        dev_labels = pickle.load(f)

    with open(directory + 'ordered_' + name + '_train_seq_feats.pkl', 'rb') as f:
        train_feats = pickle.load(f)
        # print(train_feats[:10])

    with open(directory + 'ordered_' + name + '_dev_seq_feats.pkl', 'rb') as f:
        dev_feats = pickle.load(f)
        # print(dev_feats[:10])

    return train_data, train_labels, dev_data, dev_labels, train_feats, dev_feats
    # return dev_data[:10], dev_labels[:10], dev_data[:10], dev_labels[:10], dev_feats[:10], dev_feats[:10]
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


def test(i2t, dev_data_batch, cluf_model, use_user=True, use_format=True, error_analysis=False, save_path=None):
    cluf_model.eval()
    all_predicted = []
    all_targets = []

    # selected_batch_for_output = np.random.randint(len(dev_data_batch), size=10)

    if error_analysis:
        f_write = open(save_path, "w")
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
        else:
            users, countries, days = None, None, None

        if use_format:
            data_format = batch[8]
            session = batch[9]
            client = batch[10]
            time = batch[11]
        else:
            data_format, session, client, time = None, None, None, None

        _, prob = cluf_model(words_length_tuple, pos, edges, user_info=(users, countries, days),
                             format_info=(data_format, session, client, time))

        this_predicted, this_target = manipulate_valid_pred(prob, targets)

        # print out prediction in actual word/pos tag for error analysis
        # if error_analysis and i in selected_batch_for_output:
        if error_analysis:
            idSeq2wordSeq(i2t, prob, targets, words_length_tuple, pos, edges, user_info=(users, countries, days),
                          format_info=(data_format, session, client, time), use_user=use_user, use_format=use_format,
                          f_write=f_write)
            # f_write.write("\n")

        # prob=prob.contiguous().view(-1, 1)
        # weights = targets.ne(-1)

        all_predicted = all_predicted + this_predicted
        all_targets = all_targets + this_target
        if i % 1000 == 0:
            print("testing finished " + str(i))

    # all_targets = [item for sublist in dev_labels for item in sublist]
    print("length of all_targets=all predicted is " + str(len(all_predicted)))

    if len(all_targets) < 200:
        print("all targets as " + str(all_targets))
        print("all predicted as " + str(all_predicted))

    results = evaluate_metrics(all_targets, all_predicted)
    for key in results:
        print(str(key) + ": " + str(results[key]))

    global F1
    global AUROC
    if results['F1'] > F1:
        F1 = results['F1']
    if results['auroc'] > AUROC:
        AUROC = results['auroc']
    print("The best F1 is {} and the best AUROC is {}".format(F1, AUROC))

    if error_analysis:
        f_write.close()

        # rename file with F1 appended
        os.rename(save_path, save_path + "_" + str(results['F1']))

    return results['F1']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument("--gpus", dest='gpus', default=1, type=int,
                        help="Use CUDA")
    parser.add_argument("--weight_decay", dest='weight_decay', default=0, type=int,
                        help="wight decay for adam")
    parser.add_argument("--directory", type=str, dest='directory', default='./data_es_en/ordered_feats/')
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
    # print("initial prob "+str(prob)) # maxseqlength * batch_size * 2
    #
    #
    # print(prob.data[:,:,1].contiguous())
    # print(prob.data[:, :, 1].t().contiguous())
    prob = prob.data[:, :, 1].t().contiguous().view(-1).cpu().numpy()
    # print(prob)
    # print("selected class 1 prob "+str(prob))
    # print("initial targets " + str(targets))
    # print(targets.data.contiguous().t())
    targets = targets.data.t().contiguous().view(-1).cpu().numpy()
    # print(targets)
    # print("flattened targets "+str(targets))
    # print(targets)
    # print(np.where(targets!=-1))

    valid_index = np.where(targets != -1)[0]
    return prob[valid_index].tolist(), targets[valid_index].tolist()


def segment_into_list_of_list(intervals, original_list):
    return [[original_list[i] for i in range(interval[0], interval[1])] for interval in intervals]


def idSeq2wordSeq_from_Variable_to_token(words, valid_index, i2t, key, intervals):
    words_idx = words.data.t().contiguous().view(-1).cpu().numpy()[valid_index].tolist()
    words_actual = [i2t[key][t] for t in words_idx]
    words_list_of_list = segment_into_list_of_list(intervals, words_actual)
    # [[words_actual[i] for i in range(interval[0], interval[1])] for interval in intervals]
    # print("For key: " + str(key) + "\n" + str(words_list_of_list) + "\n")
    return words_list_of_list


def idSeq2wordSeq_from_Variable_to_seqFeature(feats, i2t, key):
    this_list = [i2t[key][feat] for feat in feats.data.contiguous().view(-1).cpu().numpy()]
    # print("For key: " + str(key) + "\n" + str(this_list) + "\n")
    return this_list


def idSeq2wordSeq(i2t, prob, targets, words_length_tuple, pos, edges, user_info,
                  format_info, use_user=True, use_format=True, f_write=None):
    users, countries, days = user_info
    data_format, session, client, time = format_info
    words, lengths = words_length_tuple

    # get valid index of non-padded position
    targets = targets.data.t().contiguous().view(-1).cpu().numpy()
    prob = prob.data[:, :, 1].t().contiguous().view(-1).cpu().numpy()
    valid_index = np.where(targets != -1)[0]

    # get intervals to segment 1d sequence
    intervals = ([(sum(lengths[:i]), sum(lengths[:(i + 1)])) for i in range(len(lengths))])

    # map id to actual token (words, pos, edge_labels, etc.)

    words_list_of_list = idSeq2wordSeq_from_Variable_to_token(words, valid_index, i2t, "words", intervals)
    pos_list_of_list = idSeq2wordSeq_from_Variable_to_token(pos, valid_index, i2t, "pos", intervals)
    edges_list_of_list = idSeq2wordSeq_from_Variable_to_token(edges, valid_index, i2t, "edge_labels", intervals)

    country_list_of_list = idSeq2wordSeq_from_Variable_to_seqFeature(countries, i2t, "country")
    client_list_of_list = idSeq2wordSeq_from_Variable_to_seqFeature(client, i2t, "client")
    format_list_of_list = idSeq2wordSeq_from_Variable_to_seqFeature(data_format, i2t,
                                                                    "format")  # (one of: reverse_translate, reverse_tap, or listen; see figures above)
    session_list_of_list = idSeq2wordSeq_from_Variable_to_seqFeature(session, i2t, "session")
    user_list_of_list = idSeq2wordSeq_from_Variable_to_seqFeature(users, i2t, "user")
    time_list_of_list = idSeq2wordSeq_from_Variable_to_seqFeature(time, i2t, "time")
    days_list_of_list = idSeq2wordSeq_from_Variable_to_seqFeature(days, i2t, "days")

    target_list_of_list = segment_into_list_of_list(intervals, targets[valid_index].tolist())
    predicted_class_list_of_list = segment_into_list_of_list(intervals,
                                                             [1 if p >= 0.5 else 0 for p in prob[valid_index].tolist()])
    pred_prob_list_of_list = segment_into_list_of_list(intervals, [round(p, 5) for p in prob[valid_index].tolist()])

    # print("words" + str(words))
    print("For target " + str(target_list_of_list))
    print("For pred_prob " + str(pred_prob_list_of_list))

    list_of_sequence = list(
        zip(words_list_of_list, target_list_of_list, predicted_class_list_of_list, pred_prob_list_of_list,
            user_list_of_list, country_list_of_list, days_list_of_list, client_list_of_list, session_list_of_list,
            format_list_of_list, time_list_of_list, pos_list_of_list, edges_list_of_list))

    for sequence in list_of_sequence:
        words, targets, predicted_classes, pred_probs, user, country, day, client, session, format, time, poss, edges = sequence
        for i in range(len(words)):
            f_write.write(str(words[i]) + " " + str(targets[i]) + " " + str(predicted_classes[i]) + " " + str(
                pred_probs[i]) + " " + str(user) + " " + str(country) + " " + str(day) + " " + str(client) + " " + str(
                session) + " " + str(format) + " " + str(time) + " " + str(poss[i]) + " " + str(edges[i]) + " null\n")
        f_write.write("\n")


def load_i2t(directory='./data_es_en/ordered_feats/'):
    i2t = dict()
    for filename in os.listdir(directory):
        if "i2t" not in filename:
            continue
        key = re.search("i2t_(.*)\.pkl", filename).group(1)
        i2t[key] = pickle.load(open(directory + filename, "rb"))

    print("loaded i2t dictionaries for these keys: " + str(i2t.keys()))
    i2t["time"] = {1: "<1m", 2: "1-2m", 3: "2-3m", 4: "3-4m", 5: "4-5m", 6: "5-10m", 7: "10-15m", 8: "15-20m",
                   9: "20-25m", 10: "25-30m", 11: ">30m", 12: "null"}
    i2t["days"] = selfDict()
    return i2t


# def annealed_learning_rate(lr, epoch):
# return lr/(1+epoch)

def main():
    args = parse_arguments()

    use_user = True
    use_format = True

    timestamp = int(time_lib.time())
    output_dir = args.directory + "output/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dev_output_log_prefix = output_dir + str(timestamp) + "_"

    if use_user:
        train_data, train_labels, dev_data, dev_labels, train_feats, dev_feats = load_ordered_data(args.directory)
        i2t = load_i2t(args.directory)

    gpu_flag = int(args.gpus)
    cluf_model = CLUF(gpu_flag, args.directory)
    if gpu_flag:
        torch.cuda.set_device(0)
        cluf_model = cluf_model.cuda()

    # loss_function = nn.NLLLoss()

    parameters = cluf_model.parameters()
    learning_rate = 0.005
    batch_size = 32

    optimizer = optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = lr_scheduler.StepLR(optimizer, 3, gamma=0.1)

    train_data_batch = Dataset(train_data, train_labels, gpu_flag, batch_size, eval=False, session_feats=train_feats)
    dev_data_batch = Dataset(dev_data, dev_labels, gpu_flag, batch_size, eval=True, session_feats=dev_feats)

    # import pdb; pdb.set_trace()
    for epoch_ in range(args.num_epochs):
        sys.stdout.flush()
        print("\n\nEpoch: {}".format(epoch_))
        train_data_batch.shuffle()
        scheduler.step()

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
            else:
                users, countries, days = None, None, None

            if use_format:
                data_format = batch[8]
                session = batch[9]
                client = batch[10]
                time = batch[11]
            else:
                data_format, session, client, time = None, None, None, None

            # print("targets loaded " + str(targets))
            cluf_model.train()
            cluf_model.zero_grad()
            log_softmax_tag_scores, _ = cluf_model(words_length_tuple, pos, edges, user_info=(users, countries, days),
                                                   format_info=(data_format, session, client, time))
            loss = masked_NLLLoss(log_softmax_tag_scores, targets)

            # manipulate_valid_pred(prob, targets)

            # total_loss += loss
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print("training finished " + str(i) + " batches (batch_size 32)")
            if i % 10000 == 0:
                sys.stdout.flush()
                print("testing, after " + str(i) + " batches")
                F1 = test(i2t, dev_data_batch, cluf_model, use_user, error_analysis=False,
                          save_path=dev_output_log_prefix + "epoch" + str(epoch_) + "_" + str(i))

                # rename file with F1

        avglogloss = test(i2t, dev_data_batch, cluf_model, use_user, error_analysis=False,
                          save_path=dev_output_log_prefix + "epoch" + str(epoch_) + "_" + str(i))
        # scheduler.step(avglogloss, epoch_)
        # print(total_loss)
    return


def reshaping(data):
    data = np.array(data)
    return data.reshape(1, data.shape[0])


if __name__ == '__main__':
    main()
