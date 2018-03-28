import os
import pickle

def get_contents(ind, pos):
	added_path = "./data/pos/penntree/"
	path = added_path+str(ind)+"/"
	files = os.listdir(path)
	X = []
	Y = []
	counter = 0
	for file in files:
		with open(path+"/"+file) as f:
			
			for line in f:
				sent = []
				y = []
				counter += 1
				line = line.replace('(', '')
				line = line.replace(')', '')
				tokens = line.split()
				for word_ind in range(len(tokens)):
					if tokens[word_ind] in pos:
						y.append(tokens[word_ind])
						sent.append(tokens[word_ind+1])
				X.append(sent)
				Y.append(y)
	return X, Y

def main():
	pos = []
	with open("./data/all_pos") as f:
		for line in f:
			pos.append(line.split('\t')[1])

	train_data = []
	train_labels = []
	test_data = []
	test_labels = []

	for i in range(19):
		if i < 10:
			ind = '0'+str(i)
		else:
			ind = str(i)
		train_data = train_data + get_contents(ind, pos)[0]
		train_labels = train_labels + get_contents(ind, pos)[1]

	for i in range(22, 25):
		test_data = test_data + get_contents(ind, pos)[0]
		test_labels = test_labels + get_contents(ind, pos)[1]

	with open('./data/pos/train.txt', 'wb') as f:
		pickle.dump(train_data, f)
	with open('./data/pos/train_labels.txt', 'wb') as f:
		pickle.dump(train_labels, f)
	with open('./data/pos/test.txt', 'wb') as f:
		pickle.dump(test_data, f)
	with open('./data/pos/test_labels.txt', 'wb') as f:
		pickle.dump(test_labels, f)
	return

if __name__ == '__main__':
	main()