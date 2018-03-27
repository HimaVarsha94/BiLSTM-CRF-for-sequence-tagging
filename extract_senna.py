import pickle
import numpy as np

def senna():
	emb = open('./senna_embeddings/embeddings.txt')
	words = open('./senna_embeddings/words.lst')
	senna_dict = {}
	senna_words_list = []

	for word in words:
		senna_words_list.append(word.strip())
	a = []
	ind = 0
	for line in emb:
		nums = line.strip().split(' ')
		nums = np.array(nums)
		senna_dict[senna_words_list[ind]] = nums
		ind += 1
	with open('./senna_embeddings/senna_obj.pkl', 'w') as f:
		pickle.dump(senna_dict, f)

def senna_embeddings_matrix():
	#made for connll dataset

def main():
	senna()
	return

if __name__ == '__main__':
	main()