import cv2
import os
import sys
import random
import numpy as np
import config
import pickle

# data_rgb.py train 10 run 

if len(sys.argv) < 4:
    print 'Missing agrument'
    print 'Ex: data_rgb.py train 10 run'
    sys.exit()

# Lay tuy chon 
train = sys.argv[1]
sample_rate = int(sys.argv[2])
if sys.argv[3] == 'run':
    debug = False
else:
    debug = True

# Cau hinh folder du lieu
server = config.server()
data_input_folder = config.data_input_path()
output_path = config.data_output_path()

class_file = 'data/classInd.txt'
data_output_folder = r'{}rgb/'.format(output_path)
text_file = r'data/{}list.txt'.format(train)
out_file_folder = r'{}database/'.format(output_path)
out_file = r'{}{}-rgb{}.pickle'.format(out_file_folder,train,sample_rate)

# Tao folder chinh
if not os.path.isdir(data_output_folder):
    os.makedirs(data_output_folder) # tao data_output_folder/
    print 'Create directory ' + data_output_folder

if not os.path.isdir(out_file_folder):
    os.makedirs(out_file_folder) # tao out_file_folder/
    print 'Create directory ' + out_file_folder

count = 0
v = 0

data=[]
classInd=[]

# Tao class index tu file classInd.txt
with open(class_file) as f0:
    for line in f0:
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)


with open(text_file) as f:
    for line in f:
        # Tao duong dan va ten file anh
        arr_line = line.rstrip().split(' ')[0] # return folder/subfolder/name.mpg

        path_video = arr_line.split('/') # return array (folder,subfolder,name.mpg)
        num_name = len(path_video) # return 3
        name_video = path_video[num_name - 1].split('.')[0] # return name ko co .mpg
        folder_video = path_video[0] # return folder
        path = data_output_folder + folder_video + '/' #return data-output/folder/
        video_class = classInd.index(folder_video) # index cua video

        group = 0
        if train == 'cross':
            group = int(line.rstrip().split(' ')[2])

        # tao folder moi neu chua ton tai
        if not os.path.isdir(path):
            os.makedirs(path)
            print 'Created folder: {}'.format(folder_video)

        cap = cv2.VideoCapture(data_input_folder + arr_line)
        v += 1
        i = -1
        os.chdir(path)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
                cap.release()
                sys.exit()
            i = i + 1
            if (i%sample_rate != 0):
                continue
            
            if not debug:
                height, width, channel = frame.shape

                #random crop max size (heightxheight)
                x = int((width-height)/2)
                crop = frame[:, x:x+height].copy()

                resize_img = cv2.resize(crop, (224, 224))

                cv2.imwrite(r'{}-{}.jpg'.format(name_video, i),resize_img)

            data.append([folder_video + '/' + name_video, i, video_class, group])

            count += 1
            if (count % 1000 == 0):
                print r'Created {} samples'.format(count)

        
        # print name_video
        # Giai phong capture
        cap.release()
print 'Generate RGB: {} samples for {} dataset with {} videos'.format(count,train,v)

# Ghi du lieu dia chi ra file
with open(out_file,'wb') as f2:
    pickle.dump(data,f2)
