import cv2
import os
import sys
import random
import numpy as np
import config
import pickle

# data_seq.py train 10 run 

# Lay tuy chon 
train = sys.argv[1]
num_seq = int(sys.argv[2])
if sys.argv[3] == 'run':
    debug = False
else:
    debug = True

def render_image(pos_render, cap, path_video, rgb_pos, index):
    os.chdir(path_video)
    i = -1
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        i += 1

        # Khi i chay den vi tri duoc lay mau
        if i == pos_render:
            m = 0
            pos = i
            k1 = 10 * index
            k2 = 10 * index
            k3 = index
            while(True):
                # Tao optical flow lay mau 1
                if (m == 5):
                    prvs1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                if (m > 5) & (m <= 15):
                    if not debug:
                        next1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        flow = inst.calc(prvs1, next1, None)
                        prvs1 = next1

                        # Chuan hoa gia tri diem anh ve tu 0 den 255
                        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)

                        # Chuyen kieu ve int8
                        horz = horz.astype('uint8')
                        vert = vert.astype('uint8')

                        # Ghi anh
                        cv2.imwrite('opt1-'+str(2*k1)+'.jpg',horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        cv2.imwrite('opt1-'+str(2*k1+1)+'.jpg',vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                    k1 += 1
                # Tao optical flow lay mau 2
                if m == 0:
                    prvs2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                if (m%2 == 0) & (m != 0):
                    if not debug:
                        next2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        flow = inst.calc(prvs2, next2, None)
                
                        prvs2 = next2

                        # Chuan hoa gia tri diem anh ve tu 0 den 255
                        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)

                        # Chuyen kieu ve int8
                        horz = horz.astype('uint8')
                        vert = vert.astype('uint8')

                        # Ghi anh
                        cv2.imwrite('opt2-'+str(2*k2)+'.jpg',horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        cv2.imwrite('opt2-'+str(2*k2+1)+'.jpg',vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                    k2 += 1
                
                # Tao RGB
                if (m == rgb_pos):
                    # print m
                    if not debug:
                        height, width, channel = frame.shape

                        #random crop max size (heightxheight)
                        x = int((width-height)/2)
                        crop = frame[:, x:x+height].copy()

                        resize_img = cv2.resize(crop, (224, 224))

                        cv2.imwrite('rgb-{}.jpg'.format(k3),resize_img)

                if m >= 20:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                i += 1
                m += 1
            break

if len(sys.argv) < 4:
    print 'Missing agrument'
    print 'Ex: data_rgb.py train 3 run'
    sys.exit()

# Cau hinh folder du lieu
server = config.server()
data_input_folder = config.data_input_path()
output_path = config.data_output_path()

class_file = 'data/classInd.txt'
data_output_folder = r'{}seq{}/'.format(output_path,num_seq)
text_file = r'data/{}list.txt'.format(train)
out_file_folder = r'{}database/'.format(output_path)
out_file = r'{}{}-seq{}.pickle'.format(out_file_folder,train,num_seq)

if (len(sys.argv)) >= 5:
    split = sys.argv[4]
    text_file = r'data/{}list0{}.txt'.format(train,split)
    out_file = r'{}{}{}-seq{}.pickle'.format(out_file_folder,train,split,num_seq)

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

inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst.setUseSpatialPropagation(True)

inst1 = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst1.setUseSpatialPropagation(True)

inst2 = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst2.setUseSpatialPropagation(True)

xy = 0
with open(text_file) as f:
    for line in f:
        if xy <= 570:
            debug = True
        else:
            debug = False
        xy += 1
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

        if not os.path.isdir(path + name_video):
            os.makedirs(path + name_video) #tao data-output/foldet/name/

        cap = cv2.VideoCapture(data_input_folder + arr_line)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (length <= 60):
            pos_render = []
            # print 'Duration video {} frame(s)'.format(length)
            if (length > 20):
                pos_render = [0, length/2-10, length-21]
                for k in range(len(pos_render)):
                    rgb_pos = 10
                    if k == (len(pos_render) - 1):
                        rgb_pos = 20
                    if k == 0:
                        rgb_pos = 0
                    cap = cv2.VideoCapture(data_input_folder + arr_line)
                    render_image(pos_render[k], cap, path + name_video, rgb_pos, k)
                    cap.release()

            elif (length >= 10) & (length <= 20):
                pos_render = [0]
                rgb_render = [0, length/2, length-1]
                i = -1
                os.chdir(path + name_video)
                while(True):
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    i += 1

                    if (i == 0):
                        prvs1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                    if (i > 0) & (i <= 10):
                        if not debug:
                            next1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                            flow = inst.calc(prvs1, next1, None)
                            prvs1 = next1

                            # Chuan hoa gia tri diem anh ve tu 0 den 255
                            horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                            vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)

                            # Chuyen kieu ve int8
                            horz = horz.astype('uint8')
                            vert = vert.astype('uint8')

                            # Ghi anh
                            cv2.imwrite('opt1-'+str(2*(i-1))+'.jpg',horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            cv2.imwrite('opt1-'+str(2*(i-1)+1)+'.jpg',vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                    # Khi i chay den vi tri duoc lay mau
                    if i in rgb_render:
                        if not debug:
                            height, width, channel = frame.shape

                            #random crop max size (heightxheight)
                            x = int((width-height)/2)
                            crop = frame[:, x:x+height].copy()

                            resize_img = cv2.resize(crop, (224, 224))

                            cv2.imwrite('rgb-{}.jpg'.format(rgb_render.index(i)),resize_img)
                cap.release()
            else:
                continue

            data.append([folder_video + '/' + name_video, pos_render, video_class, group])
            count += 1
            v += 1
            
        else:
            pos_render = []

            divide = length / num_seq
            # print(length, divide)

            for i in range(num_seq):
                if i < num_seq - 1:
                    k = np.random.randint(divide*i,divide*(i+1)-19)
                else:
                    k = np.random.randint(divide*i,length-20)
                pos_render.append(k)

            v += 1
            i = -1
            os.chdir(path + name_video)
            while(True):
                # Capture frame-by-frame
                if i not in pos_render:
                    ret, frame = cap.read()
                    if not ret:
                        print('Break,' i)
                        break
                    i += 1

                # Khi i chay den vi tri duoc lay mau
                if i in pos_render:
                    m = 0
                    pos = i
                    k1 = 10 * pos_render.index(i)
                    k2 = 10 * pos_render.index(i)
                    k3 = pos_render.index(i)
                    while(True):
                        # Tao optical flow lay mau 1
                        if (m == 5):
                            prvs1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                        if (m > 5) & (m <= 15):
                            if not debug:
                                next1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                                flow = inst1.calc(prvs1, next1, None)
                                prvs1 = next1

                                # Chuan hoa gia tri diem anh ve tu 0 den 255
                                horz1 = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                                vert1 = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)

                                # Chuyen kieu ve int8
                                horz1 = horz1.astype('uint8')
                                vert1 = vert1.astype('uint8')

                                # Ghi anh
                                cv2.imwrite('opt1-'+str(2*k1)+'.jpg',horz1,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                cv2.imwrite('opt1-'+str(2*k1+1)+'.jpg',vert1,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                            k1 += 1
                        # Tao optical flow lay mau 2
                        if m == 0:
                            prvs2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                        if (m%2 == 0) & (m != 0):
                            if not debug:
                                next2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                                flow = inst2.calc(prvs2, next2, None)
                        
                                prvs2 = next2

                                # Chuan hoa gia tri diem anh ve tu 0 den 255
                                horz2 = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                                vert2 = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)

                                # Chuyen kieu ve int8
                                horz2 = horz2.astype('uint8')
                                vert2 = vert2.astype('uint8')

                                # Ghi anh
                                cv2.imwrite('opt2-'+str(2*k2)+'.jpg',horz2,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                cv2.imwrite('opt2-'+str(2*k2+1)+'.jpg',vert2,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                            k2 += 1
                        
                        # Tao RGB
                        if (m == 10):
                            if not debug:
                                height, width, channel = frame.shape

                                #random crop max size (heightxheight)
                                x = int((width-height)/2)
                                crop = frame[:, x:x+height].copy()

                                resize_img = cv2.resize(crop, (224, 224))

                                cv2.imwrite('rgb-{}.jpg'.format(k3),resize_img)

                        if m >= 20:
                            break

                        ret, frame = cap.read()
                        if not ret:
                            print('Break,' i)
                            break
                        i += 1
                        m += 1

            if (k1 == 30) & (k2 == 30) & (k3 == 2):
                count += 1
                data.append([folder_video + '/' + name_video, pos_render, video_class, group])
            else:
                print (k1, k2, k3, name_video)
                print pos_render
                print length
                sys.exit()

        if (count % 100 == 0):
            print r'Created {} samples'.format(count)

        
        # print name_video
        # Giai phong capture
        cap.release()
print 'Generate {} samples for {} dataset'.format(len(data),train)

# Ghi du lieu dia chi ra file
with open(out_file,'wb') as f2:
    pickle.dump(data,f2)
