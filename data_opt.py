import cv2
import os
import sys
import random
import numpy as np
import pickle
import config

# data_opt.py train 1 run

if len(sys.argv) < 4:
    print 'Missing agrument'
    print 'Ex: data_opt.py train 1 run'
    sys.exit()

train = sys.argv[1]
sample_rate = int(sys.argv[2])
if sample_rate == 1:
    opt_rate = 10
else:
    opt_rate = 5

if sys.argv[3] == 'run':
    debug = False
else: 
    debug = True

# Cau hinh duong dan
server = config.server()
data_input_folder = config.data_input_path()
output_path = config.data_output_path()

class_file = 'data/classInd.txt'
data_output_folder = r'{}opt{}/'.format(output_path,sample_rate)
text_file = r'data/{}list.txt'.format(train)
out_file_folder = r'{}database/'.format(output_path)
out_file = r'{}{}-opt{}.pickle'.format(out_file_folder,train,sample_rate)
    
data=[]
classInd=[]

# Tao folder chinh
if not os.path.isdir(data_output_folder):
    os.makedirs(data_output_folder) # tao data_output_folder/
    print 'Create directory ' + data_output_folder

if not os.path.isdir(out_file_folder):
    os.makedirs(out_file_folder) # tao out_file_folder/
    print 'Create directory ' + out_file_folder

# Tao class index tu file classInd.txt
with open(class_file) as f0:
    for line in f0:
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)

# Khoi tao tinh optical flow bang DIS-Fast
c = 0
inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst.setUseSpatialPropagation(True)

with open(text_file) as f1:
    for line in f1:
        # tao ten va folder anh
        arr_line = line.rstrip().split(' ')[0]
        path_video = arr_line.split('/') # return array (folder,subfolder,name.mpg)
        num_name = len(path_video) # return 3
        name_video = path_video[num_name - 1].split('.')[0] # return name ko co .mpg
        folder_video = path_video[0] # return folder
        path = data_output_folder + folder_video + '/' # return data-output/folder/
        video_class = classInd.index(folder_video) # index cua video

        group = 0
        if train == 'cross':
            group = int(line.rstrip().split(' ')[2])

        if not os.path.isdir(path):
            os.makedirs(path) # tao data-ouput/folder/
            print 'Start with ' + path

        if not os.path.isdir(path + name_video):
            os.makedirs(path + name_video) #tao data-output/foldet/name/
            # print 'make dir ' + path + name_video

        cap = cv2.VideoCapture(data_input_folder + arr_line)
        ret, frame1 = cap.read()
        if not ret:
            print 'Can\'t read ' + data_input_folder + arr_line
            continue

        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        k = 0
        m = 0
        os.chdir(path + name_video)
        while(True):  
            # Capture frame-by-frame
            ret, frame2 = cap.read()
            if not ret:
                break;

            if m%sample_rate == 0:

                if (k%opt_rate == 0) & (k > 9):
                    data.append([folder_video + '/' + name_video, 2*(k-10), video_class, group])
                    c+=1
                    if (c % 1000 == 0):
                        print r'Created {} samples'.format(c)
                if not debug:
                    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                    
                    # flow = optical_flow.calc(prvs, next, None)
                    # flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow = inst.calc(prvs, next, None)
            
                    prvs = next

                    # Chuan hoa gia tri diem anh ve tu 0 den 255
                    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)

                    # Chuyen kieu ve int8
                    horz = horz.astype('uint8')
                    vert = vert.astype('uint8')

                    # Ghi anh
                    cv2.imwrite(str(2*k)+'.jpg',horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    cv2.imwrite(str(2*k+1)+'.jpg',vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                k+=1

            m+=1

        if ((k%opt_rate > int(opt_rate/2)) | (k%opt_rate == 0)) & (k > 9):
            data.append([folder_video + '/' + name_video, 2*(k-10), video_class, group])
            c+=1
            if (c % 1000 == 0):
                print r'Created {} samples'.format(c)

        # print name_video

        # Giai phong capture
        cap.release()

print 'Generate opt{}: {} samples for {} dataset'.format(sample_rate,c,train)

# Ghi du lieu dia chi ra file
with open(out_file,'wb') as f2:
    pickle.dump(data,f2)
