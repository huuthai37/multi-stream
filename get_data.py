import numpy as np
import sys
import pickle
import random
from PIL import Image
import cv2
from keras.utils import np_utils
import config
from sklearn.metrics import classification_report
from keras import backend as K
import math

server = config.server()
data_output_path = config.data_output_path()
data_folder_rgb = r'{}rgb/'.format(data_output_path)
data_folder_seq = r'{}seq3/'.format(data_output_path)

def getTrainData(keys,batch_size,classes,mode,train,opt_size,seq=False): 
    """
    mode 1: Single Stream
    mode 2: Two Stream
    mode 3: Multiple Stream
    """
    while 1:
        for i in range(0, len(keys), batch_size):
            if not seq:
                if mode == 1:
                    X_train, Y_train = stack_single_stream(
                        chunk=keys[i:i+batch_size],
                        opt_size=opt_size,
                        batch_size=batch_size)

                elif mode == 2:
                    X_train, Y_train = stack_two_stream(
                        chunk=keys[i:i+batch_size],
                        multi_opt_size=opt_size,
                        batch_size=batch_size)

                else:
                    X_train, Y_train=stack_multi_stream(
                        chunk=keys[i:i+batch_size],
                        multi_opt_size=opt_size,
                        batch_size=batch_size)
            else:
                if mode == 1:
                    X_train, Y_train = stack_single_seq(
                        chunk=keys[i:i+batch_size],
                        opt_size=opt_size,
                        batch_size=batch_size)

                elif mode == 2:
                    X_train, Y_train = stack_two_seq(
                        chunk=keys[i:i+batch_size],
                        multi_opt_size=opt_size,
                        batch_size=batch_size)

                else:
                    X_train, Y_train=stack_multi_seq(
                        chunk=keys[i:i+batch_size],
                        multi_opt_size=opt_size,
                        batch_size=batch_size)

            Y_train = np_utils.to_categorical(Y_train,classes)
            if train == 'test':
                print 'Test batch {}'.format(i/batch_size+1)
            yield X_train, np.array(Y_train)

def getClassData(keys,cut=0):
    labels = []
    if cut == 0:
        for opt in keys:
            labels.append(opt[2])
    else:
        i = 0
        for opt in keys:
            labels.append(opt[2])
            i += 1
            if i >= cut:
                break

    return labels

def getScorePerVideo(result, data):
    indVideo = []
    dataVideo = []
    length = len(data)
    for i in range(length):
        name = data[i][0].split('/')[1]
        if name not in indVideo:
            indVideo.append(name)
            dataVideo.append([name,data[i][2],result[i], 1])
        else:
            index = indVideo.index(name)
            dataVideo[index][2] = dataVideo[index][2] + result[i]
            dataVideo[index][3] += 1

    resultVideo = []
    classVideo = []
    len_data = len(dataVideo)
    for i in range(len_data):
        pred = dataVideo[i][2] / dataVideo[i][3]
        resultVideo.append(pred)
        classVideo.append(dataVideo[i][1])

    resultVideoArr = np.array(resultVideo)
    classVideoArr = np.array(classVideo)

    y_classes = resultVideoArr.argmax(axis=-1)
    return (classification_report(classVideoArr, y_classes, digits=6))

def get_data_cross_validation(keys_cross,cross_index):
    length = len(keys_cross)
    train_data = []
    test_data = []
    for i in range(length):
        if keys_cross[i][3] != cross_index:
            train_data.append(keys_cross[i])
        else:
            test_data.append(keys_cross[i])

    return train_data, test_data


def stack_rgb(sample,start_rgb):
    folder_rgb = sample[0]
    rgb = cv2.imread(data_folder_rgb + folder_rgb + '-' + str(start_rgb) + '.jpg')
    rgb = rgb.astype('float16',copy=False)
    rgb/=255
    return rgb

def stack_optical_flow(sample,start_opt,data_folder):
    if not server:
        return np.zeros((224,224,20))
    folder_opt = sample[0] + '/'
    arrays = []

    for i in range(start_opt, start_opt + 20):
        img = cv2.imread(data_folder + folder_opt + str(i) + '.jpg', 0)
        if img is None:
            print data_folder + folder_opt
            sys.exit()
        height, width = img.shape
        crop_pos = int((width-height)/2)
        img = img[:,crop_pos:crop_pos+height]
        resize_img = cv2.resize(img, (224, 224))

        resize_img = resize_img.astype('float16',copy=False)
        resize_img/=255
        opt_nor = resize_img - resize_img.mean()

        arrays.append(opt_nor)

    nstack = np.dstack(arrays)

    return nstack

def stack_single_stream(chunk,opt_size,batch_size):
    labels = []
    stack_return = []
    if opt_size[0] == 0:
        for rgb in chunk:
            labels.append(rgb[2])
            start_rgb = rgb[1]
            stack_return.append(stack_rgb(rgb,start_rgb))
    else:
        data_folder_opt = r'{}opt{}/'.format(data_output_path,opt_size[0])
        for opt in chunk:
            labels.append(opt[2])
            start_opt = opt[1]
            stack_return.append(stack_optical_flow(opt,start_opt,data_folder_opt))

    if len(stack_return) < len(chunk):
        print 'Stacked data error'
        sys.exit()

    return (np.array(stack_return), labels)

def stack_two_stream(chunk,multi_opt_size,batch_size):
    labels = []
    stack_rgbs = []
    stack_opt = []

    opt_bare = multi_opt_size[1]
    for sample in chunk:
        labels.append(sample[2])

        for opt_size in multi_opt_size:
            if opt_size == 0:
                start_opt = sample[1]
                if (start_opt % 20 > 0):
                    start_rgb = (int(np.floor(start_opt * opt_bare / 20)) + 1 ) * 10
                else:
                    start_rgb = int(start_opt * opt_bare / 2)
                stack_rgbs.append(stack_rgb(sample,start_rgb))
            else:
                opt_bare = opt_size
                start_opt = sample[1]
                data_folder_opt = r'{}opt{}/'.format(data_output_path,opt_size)
                stack_opt.append(stack_optical_flow(sample,start_opt,data_folder_opt))

    if len(stack_opt) < len(chunk):
        print 'Stacked data error'
        sys.exit()

    return [np.array(stack_rgbs), np.array(stack_opt)], labels

def stack_multi_stream(chunk,multi_opt_size,batch_size):
    labels = []
    returns = []
    stack_return = []

    if len(chunk[0]) < 7:
        print 'Input chunk error'
        sys.exit()

    if 3 in multi_opt_size:
        print 'Optical flow sample rate 3 not support!!!'
        sys.exit()

    for opt_size in multi_opt_size:
        stack_return.append([])

    for sample in chunk:
        labels.append(sample[2])

        s = 0
        for opt_size in multi_opt_size:
            if opt_size == 0:
                start_rgb = sample[1]
                stack_return[s].append(stack_rgb(sample,start_rgb))
            else:
                start_opt = sample[4+int(math.log(opt_size,2))]
                data_folder_opt = r'{}opt{}/'.format(data_output_path,opt_size)
                stack_return[s].append(stack_optical_flow(sample,start_opt,data_folder_opt))
            s+=1

    if len(stack_return[0]) < len(chunk):
        print 'Stacked data error'
        sys.exit()

    for i in range(len(multi_opt_size)):
        returns.append(np.array(stack_return[i]))

    return returns, labels

def stack_seq_rgb(path_video):
    return_stack = []
    for i in range(3):
        img = cv2.imread(data_folder_seq + path_video + '/rgb-' + str(i) + '.jpg')
        rgb = cv2.resize(img, (299, 299))
        rgb = rgb.astype('float16',copy=False)
        rgb/=255
        return_stack.append(rgb)
    return np.array(return_stack)

def stack_seq_optical_flow(path_video,render_opt,opt_size):
    # if not server:
    #     return np.zeros((3,224,224,20))
    arrays = []
    return_data = []
    if len(render_opt) == 3:
        for k in range(3):
            for i in range(k*20 + 5, k*20 + 15):
                img = cv2.imread(data_folder_seq + path_video + '/opt' + str(opt_size) + '-' + str(i) + '.jpg', 0)
                if img is None:
                    print 'Error render optical flow'
                    sys.exit()
                height, width = img.shape
                crop_pos = int((width-height)/2)
                img = img[:,crop_pos:crop_pos+height]
                resize_img = cv2.resize(img, (299, 299))

                resize_img = resize_img.astype('float16',copy=False)
                resize_img/=255
                opt_nor = resize_img - resize_img.mean()

                arrays.append(opt_nor)

            nstack = np.dstack(arrays)
            arrays = []
            return_data.append(nstack)
    else:
        for i in range(5,15):
            img = cv2.imread(data_folder_seq + path_video + '/opt1-' + str(i) + '.jpg', 0)
            if img is None:
                print 'Error render optical flow'
                sys.exit()
            height, width = img.shape
            crop_pos = int((width-height)/2)
            img = img[:,crop_pos:crop_pos+height]
            resize_img = cv2.resize(img, (299, 299))

            resize_img = resize_img.astype('float16',copy=False)
            resize_img/=255
            opt_nor = resize_img - resize_img.mean()

            arrays.append(opt_nor)

        nstack = np.dstack(arrays)
        return_data.append(nstack)
        return_data.append(nstack)
        return_data.append(nstack)

    return (return_data)

def stack_single_seq(chunk,opt_size,batch_size):
    labels = []
    stack_return = []
    if opt_size[0] == 0:
        for rgb in chunk:
            labels.append(rgb[2])
            stack_return.append(stack_seq_rgb(rgb[0]))
    else:
        for opt in chunk:
            labels.append(opt[2])
            stack_return.append(stack_seq_optical_flow(opt[0],opt[1],opt_size[0]))

    if len(stack_return) < len(chunk):
        print 'Stacked data error'
        sys.exit()

    return np.array(stack_return), labels

def stack_multi_seq(chunk,multi_opt_size,batch_size):
    labels = []
    returns = []
    stack_return = []

    for opt_size in multi_opt_size:
        stack_return.append([])

    for sample in chunk:
        labels.append(sample[2])

        s = 0
        for opt_size in multi_opt_size:
            if opt_size == 0:
                stack_return[s].append(stack_seq_rgb(sample[0]))
            else:
                stack_return[s].append(stack_seq_optical_flow(sample[0],sample[1],opt_size))
            s+=1

    if len(stack_return[0]) < len(chunk):
        print 'Stacked data error'
        sys.exit()

    for i in range(len(multi_opt_size)):
        returns.append(np.array(stack_return[i]))

    return returns, labels

def convert_weights(weights, depth, size=3, ins=32):
    mat = weights[0]
    mat2 = np.empty([size,size,depth,ins])
    for i in range(ins):
        x=(mat[:,:,0,i] + mat[:,:,1,i] + mat[:,:,2,i])/3
        for j in range(depth):
            mat2[:,:,j,i] = x
    return [mat2]

def get_model_memory_usage(model, batch_size):

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes



