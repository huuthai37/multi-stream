import keras
import sys
import time
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import TimeDistributed
from keras.layers import LSTM
import get_data as gd
from keras import optimizers
import pickle
import random
import numpy as np
import config
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import mobilenet

# temporal_stream.py [train|retrain|test]{+cross} {batch} {classes} {epochs} {sample rate} {sequence length} {old epochs} {cross index}
if len(sys.argv) < 6:
    print 'Missing agrument'
    print r'Ex: temporal_stream.py train {batch} {classes} {epochs} {sample rate} {sequence length}'
    sys.exit()

process = sys.argv[1].split('+')
if process[0] == 'train':
    train = True
    retrain = False
    old_epochs = 0
elif process[0] == 'retrain':
    if len(sys.argv) < 7:
        print 'Missing agrument'
        print r'Ex: temporal_stream.py retrain {batch} {classes} {epochs} {sample rate} {sequence length} {old epochs}'
        sys.exit()
    train = True
    retrain = True
    old_epochs = int(sys.argv[7])
else:
    train = False
    retrain = False

batch_size = int(sys.argv[2])
classes = int(sys.argv[3])
epochs = int(sys.argv[4])
opt_size = int(sys.argv[5])
seq_len = int(sys.argv[6])
n_neurons = 100

depth = 20
input_shape = (224,224,depth)

server = config.server()
data_output_path = config.data_output_path()

cross_index = 0
cross_validation = False
if (len(process) > 1) :
    if (process[1] == 'cross'):
        cross_validation = True
        cross_index = int(sys.argv[len(sys.argv)-1])

if not cross_validation:
    if train:
        out_file = r'{}database/train-seq{}.pickle'.format(data_output_path,seq_len)
        valid_file = r'{}database/test-seq{}.pickle'.format(data_output_path,seq_len)
    else:
        out_file = r'{}database/test-seq{}.pickle'.format(data_output_path,seq_len)
else:
    if train:
        out_file = r'{}database/train{}-seq{}.pickle'.format(data_output_path,cross_index,seq_len)
        valid_file = r'{}database/test{}-seq{}.pickle'.format(data_output_path,cross_index,seq_len)
    else:
        out_file = r'{}database/test{}-seq{}.pickle'.format(data_output_path,cross_index,seq_len)

# MobileNet model
if train & (not retrain):
    mobilenet = mobilenet.mobilenet_remake(
        name='temporal',
        input_shape=(224,224,depth),
        classes=classes,
        weight='imagenet'
    )
else:
    mobilenet = mobilenet.mobilenet_remake(
        name='temporal',
        input_shape=(224,224,depth),
        classes=classes,
        weight=None
    )

result_model = Sequential()
result_model.add(TimeDistributed(mobilenet, input_shape=(None,224,224,depth)))
result_model.add(LSTM(n_neurons, return_sequences=True))
result_model.add(LSTM(n_neurons, return_sequences=False))
result_model.add(Dropout(0.25))
result_model.add(Dense(classes, activation='softmax'))

# Run
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/temporal_lstm{}_{}e_cr{}.h5'.format(opt_size,old_epochs,cross_index))


    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)

    with open(valid_file,'rb') as f2:
        keys_valid = pickle.load(f2)
    len_valid = len(keys_valid)

#         with open(out_file,'rb') as f1:
#             keys_cross = pickle.load(f1)
#         keys, keys_valid = gd.get_data_cross_validation(keys_cross,cross_index)

#         len_samples = len(keys)
#         len_valid = len(keys_valid)

    print('-'*40)
    print 'MobileNet Temporal{} stream: Training'.format(opt_size)
    print 'Number samples: {}'.format(len_samples)
    print 'Number valid: {}'.format(len_valid)
    print('-'*40)

    histories = []
    if server:
        steps = len_samples/batch_size
        validation_steps = int(np.ceil(len_valid*1.0/batch_size))
    else:
        steps = 5
        validation_steps = 5
    
    for e in range(epochs):
        print('Epoch', e+1)
        print('-'*40)

        if server:
            random.shuffle(keys)

        time_start = time.time()

        history = result_model.fit_generator(
            gd.getTrainData(
                keys=keys,batch_size=batch_size,classes=classes,mode=1,train='train',opt_size=[opt_size],seq=True), 
            verbose=1, 
            max_queue_size=3, 
            steps_per_epoch=steps, 
            epochs=1,
            validation_data=gd.getTrainData(
                keys=keys_valid,batch_size=batch_size,classes=classes,mode=1,train='test',opt_size=[opt_size],seq=True),
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
        result_model.save_weights('weights/temporal_lstm{}_{}e_cr{}.h5'.format(opt_size,old_epochs+1+e,cross_index))

        with open('histories/temporal_lstm{}_{}_{}e_cr{}'.format(opt_size, old_epochs, epochs,cross_index), 'wb') as file_pi:
            pickle.dump(histories, file_pi)

else:
    result_model.load_weights('weights/temporal_lstm{}_{}e_cr{}.h5'.format(opt_size,epochs,cross_index))

    # if not cross_validation:
    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    # else:
    #     with open(out_file,'rb') as f1:
    #         keys_cross = pickle.load(f1)
    #     keys_train, keys = gd.get_data_cross_validation(keys_cross,cross_index)

    len_samples = len(keys)

    print('-'*40)
    print 'MobileNet Temporal{} stream: Testing'.format(opt_size)
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)

    if server:
        Y_test = gd.getClassData(keys)
        steps = int(np.ceil(len_samples*1.0/batch_size))
    else:
        Y_test = gd.getClassData(keys, 10*batch_size)
        steps = 10

    time_start = time.time()

    y_pred = result_model.predict_generator(
        gd.getTrainData(
            keys=keys,batch_size=batch_size,classes=classes,mode=1,train='test',opt_size=[opt_size],seq=True), 
        max_queue_size=3, 
        steps=steps)

    run_time = time.time() - time_start
    
    y_classes = y_pred.argmax(axis=-1)
    print 'Score per samples'
    print(classification_report(Y_test, y_classes, digits=6))
    with open('results/temporal-lstm-{}-cr{}.txt'.format(opt_size,cross_index), 'w+') as fw1:
        fw1.write(classification_report(Y_test, y_classes, digits=6))
        fw1.write('\nRun time: ' + str(run_time))

    print 'Confusion matrix'
    print(confusion_matrix(Y_test, y_classes))
    with open('results/temporal-lstm-{}-cf-cr{}.txt'.format(opt_size,cross_index),'wb') as fw3:
        pickle.dump(confusion_matrix(Y_test, y_classes),fw3)

    with open('results/temporal-lstm-{}-cr{}.pickle'.format(opt_size,cross_index),'wb') as fw3:
        pickle.dump([y_pred, Y_test],fw3)


    print 'Run time: {}'.format(run_time)
