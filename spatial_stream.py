import keras
import sys
import time
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten, Input, ZeroPadding2D, Dropout
import get_data as gd
from keras import optimizers
import pickle
import random
import numpy as np
import config
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# spatial_stream.py [train|retrain|test] {batch} {classes} {epochs} {sample rate} {old epochs}
if len(sys.argv) < 6:
    print 'Missing agrument'
    print r'Ex: spatial_stream.py train {batch} {classes} {epochs} {sample rate}'
    sys.exit()

if sys.argv[1] == 'train':
    train = True
    retrain = False
    old_epochs = 0
elif sys.argv[1] == 'retrain':
    if len(sys.argv) < 7:
        print 'Missing agrument'
        print r'Ex: spatial_stream.py retrain {batch} {classes} {epochs} {sample rate} {old epochs}'
        sys.exit()
    train = True
    retrain = True
    old_epochs = int(sys.argv[6])
else:
    train = False
    retrain = False

batch_size = int(sys.argv[2])
classes = int(sys.argv[3])
epochs = int(sys.argv[4])
sample_rate = int(sys.argv[5])

server = config.server()
data_output_path = config.data_output_path()

if train:
    out_file = r'{}database/train-rgb{}.pickle'.format(data_output_path,sample_rate)
    valid_file = r'{}database/test-rgb{}.pickle'.format(data_output_path,sample_rate)
else:
    out_file = r'{}database/test-rgb{}.pickle'.format(data_output_path,sample_rate)

# MobileNet model
if train & (not retrain):
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        weights='imagenet',
        dropout=0.5
    )
else:
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        dropout=0.5
    )

# Modify network some last layer
x = Flatten()(model.layers[-4].output)
x = Dense(classes, activation='softmax', name='predictions')(x)

#Then create the corresponding model 
result_model = Model(inputs=model.input, outputs=x)
# result_model.summary()
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/spatial_{}e.h5'.format(old_epochs))

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)

    with open(valid_file,'rb') as f2:
        keys_valid = pickle.load(f2)
    len_valid = len(keys_valid)

    print('-'*40)
    print('Spatial stream only: Training')
    print 'Number samples: {}'.format(len_samples)
    print 'Number valid: {}'.format(len_valid)
    print('-'*40)

    histories = []
    if server:
        steps = len_samples/batch_size
        validation_steps = int(np.ceil(len_valid*1.0/batch_size))
    else:
        steps = 10
        validation_steps = 10
    
    for e in range(epochs):
        print('Epoch', e+1)
        print('-'*40)

        if server:
            random.shuffle(keys)

        time_start = time.time()

        history = result_model.fit_generator(
            gd.getTrainData(
                keys=keys,batch_size=batch_size,classes=classes,mode=1,train='train',opt_size=[0]), 
            verbose=1, 
            max_queue_size=3, 
            steps_per_epoch=steps, 
            epochs=1,
            validation_data=gd.getTrainData(
                keys=keys_valid,batch_size=batch_size,classes=classes,mode=1,train='test',opt_size=[0]),
            validation_steps=validation_steps
        )
        run_time = time.time() - time_start

        histories.append([
            history.history['acc'],
            history.history['val_acc'],
            history.history['loss'],
            history.history['val_loss'],
            run_time
        ])
        result_model.save_weights('weights/spatial_{}e.h5'.format(old_epochs+1+e))

        with open('histories/spatial{}_{}_{}e'.format(sample_rate,old_epochs,epochs), 'wb') as file_pi:
            pickle.dump(histories, file_pi)

else:
    result_model.load_weights('weights/spatial_{}e.h5'.format(epochs))

    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)

    print('-'*40)
    print 'Spatial stream: Testing'
    print 'Number samples: {}'.format(len_samples)
    print('-'*40)

    if server:
        Y_test = gd.getClassData(keys)
        steps = int(np.ceil(len_samples*1.0/batch_size))
    else:
        Y_test = gd.getClassData(keys, 10*batch_size)
        steps = 10
        random.shuffle(keys)

    time_start = time.time()

    y_pred = result_model.predict_generator(
        gd.getTrainData(
            keys=keys,batch_size=batch_size,classes=classes,mode=1,train='test',opt_size=[0]), 
        max_queue_size=3, 
        steps=steps)

    run_time = time.time() - time_start
    
    y_classes = y_pred.argmax(axis=-1)
    print 'Score per samples'
    print(classification_report(Y_test, y_classes, digits=6))
    with open('results/spatial.txt', 'w+') as fw1:
        fw1.write(classification_report(Y_test, y_classes, digits=6))
        fw1.write('\nRun time: ' + str(run_time))

    if server:
        print 'Score per video'
        print(gd.getScorePerVideo(y_pred, keys))
        with open('results/spatial-v.txt', 'w+') as fw2:
            fw2.write(gd.getScorePerVideo(y_pred, keys))
    else:
        print 'Score per video'
        print(gd.getScorePerVideo(y_pred, keys[0:10*batch_size]))
        with open('results/spatial-v.txt', 'w+') as fw2:
            fw2.write(gd.getScorePerVideo(y_pred, keys[0:10*batch_size]))

    print 'Confusion matrix'
    print confusion_matrix(Y_test, y_classes)
    with open('results/spatial-cf.txt','wb') as fw3:
        pickle.dump(confusion_matrix(Y_test, y_classes),fw3)

    print 'Run time: {}'.format(run_time)
