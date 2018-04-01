import pickle
import numpy as np

def main():
	mode = "POS"

	if mode == "NER":
		filename = "result/test_ner_bilstm_cnn4277.txt"
	elif mode=="POS":
		filename = "result/test_pos_bilstm_cnn9000.txt"

	with open(filename,"r") as f:
		text=[]
		prediction=[]
		groundtruth=[]

		flag=False
		# we only output interesting case, either groundtruth has mistakes or our prediction has mistakes
		for line in f.readlines():


			if (len(line)<5):
				if (flag):
					print(text)
					print(prediction)
					print(groundtruth)
				flag=False
				text = []
				prediction = []
				groundtruth = []
			else:
				text+=[line.strip().split(" ")[0]]
				prediction+=[line.strip().split(" ")[1]]
				groundtruth+=[line.strip().split(" ")[2]]
				if line.strip().split(" ")[1]!=line.strip().split(" ")[2]:
					flag=True



if __name__ == '__main__':
	main()