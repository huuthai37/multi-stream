import numpy as np
import pickle
import sys
import cv2
import config

# check_seq.py train 10 

# Lay tuy chon 
train = sys.argv[1]
num_seq = int(sys.argv[2])

server = config.server()
output_path = config.data_output_path()

data_output_folder = r'{}seq{}/'.format(output_path,num_seq)
out_file_folder = r'{}database/'.format(output_path)
out_file = r'{}{}-seq{}.pickle'.format(out_file_folder,train,num_seq)

if (len(sys.argv)) >= 4:
    split = sys.argv[3]
    out_file = r'{}{}{}-seq{}.pickle'.format(out_file_folder,train,split,num_seq)

with open(out_file,'rb') as f1:
    keys = pickle.load(f1)

length = len(keys)
print ('Video:', length)

for i in range(length):
	fileimg = keys[i][0] + '/'
	if (len(keys[i][1]) == 3):
		for j in range(60):
			img = cv2.imread(data_output_folder + fileimg + 'opt1-' + str(j) + '.jpg')
			if (img is None):
				print ('opt1', fileimg)
				break
		for j in range(60):
			img = cv2.imread(data_output_folder + fileimg + 'opt2-' + str(j) + '.jpg')
			if (img is None):
				print ('opt2', fileimg)
				break
	else:
		for j in range(20):
			img = cv2.imread(data_output_folder + fileimg + 'opt1-' + str(j) + '.jpg')
			if (img is None):
				print ('opt1', fileimg)
				break
	for j in range(3):
		img = cv2.imread(data_output_folder + fileimg + 'rgb-' + str(j) + '.jpg')
		if (img is None):
			print ('rgb', fileimg)
			break
	if i % 100 == 0:
		print ('Checked', i)
