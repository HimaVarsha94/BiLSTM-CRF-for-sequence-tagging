import pickle

def main():
	for epoch in range(1):
		with open('./data/duolingo/data_labels.txt', 'rb') as f:
			all_data_labels = pickle.load(f)
		with open('./data/duolingo/data.txt', 'rb') as f:
			all_data = pickle.load(f)
		length = len(all_data_labels)
		targets = all_data_labels[int(length*0.9):]
		all_data = all_data[int(length*0.9):]
		with open('./duolingo_models/results'+str(epoch), 'rb') as f:
				predicted_values = pickle.load(f)
		with open('./duolingo_models/output.txt', 'w') as f:
			for ind in range(len(targets)):
				length_sent = len(targets[ind])
				for word_id in range(length_sent):
					try:
						f.write(all_data[ind][word_id]+'\t'+str(predicted_values[ind][word_id])+'\t'+str(targets[ind][word_id]) +'\n')
					except:
						import pdb; pdb.set_trace()
				f.write('\n')



if __name__ == '__main__':
	main()