import argparse, sys

#for preprocess the conlll2000 data into lists
def chunking_preprocess(datafile, senna=True):
	counter = 0
	data = []
	X = []
	y = []
	if senna == True:
		senna_dict = pickle.load("./senna_embeddings/senna_obj.pkl")
	for line in datafile:
		line = line.strip()
		tokens = line.split(' ')

		if len(tokens) == 1:
			counter += 1
			X.append(new_data)
			new_data = []
		else:
			new_data.append(senna_dict[tokens[0]])
			y.append(tokens[2])
			new_data.append((tokens[0], tokens[1], tokens[2]))
	return X, y

#loads and preprocess the conll data
def load_chunking():
	train_data = open('./data/conll2000/train.txt')
	test_data = open('./data/conll2000/test.txt')

	X_train, y_train = chunking_preprocess(train_data)
	X_test, y_test = chunking_preprocess(test_data)
	return training_data, testing_data

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument("--d", dest="dataset", default=1, help="POS(0), CONLL2000(1), CONLL2003(2)", type=int)
	parser.add_argument("--v", dest="verbose", default=1, help="verbose", type=int)
	return parser.parse_args()

def main(args):
	if args.dataset == 1:
		load_chunking()
	return

if __name__ == '__main__':
	rg = parse_arguments()
	main(rg)
