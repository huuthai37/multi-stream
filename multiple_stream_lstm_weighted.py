import keras
import sys
import time
from keras.models import Model
from keras.layers import Dense, Conv1D, Conv2D, Activation, Reshape, Flatten, Input, ZeroPadding2D, Dropout
from keras.layers import Average, Multiply, Maximum, GlobalAveragePooling2D, Concatenate,TimeDistributed, LSTM
import get_data as gd
from keras import optimizers
import pickle
import random
import numpy as np
import config
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import mobilenet

# multiple_stream_lstm_weighted.py [train|retrain|test]{+cross} {batch} {classes} {epochs} {fusion} [list 4 single epochs] {cross index}
if len(sys.argv) < 6:
    print 'Missing agrument'
    print r'Ex: multiple_stream_lstm_weighted.py train {batch} {classes} {epochs} {fusion} {0_0_0_0}'
    sys.exit()

process = sys.argv[1].split('+')
if process[0] == 'train':
    train = True
    retrain = False
    old_epochs = int(sys.argv[7])
elif process[0] == 'retrain':
    if len(sys.argv) < 6:
        print 'Missing agrument'
        print r'Ex: multiple_stream_lstm_weighted.py retrain {batch} {classes} {epochs} {sample rate} {old epochs}'
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
fusion = sys.argv[5]
opt_size = sys.argv[6]
n_neurons = 100

multi_opt_size = []
pretrains = []
arr_opt_size = opt_size.split('_')
for i in range(len(arr_opt_size)):
    if arr_opt_size[i] != '0':
        multi_opt_size.append(i)
        pretrains.append(int(arr_opt_size[i]))

print multi_opt_size
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
        out_file = r'{}database/train-seq3.pickle'.format(data_output_path)
        valid_file = r'{}database/test-seq3.pickle'.format(data_output_path)
    else:
        out_file = r'{}database/test-seq3.pickle'.format(data_output_path)
else:
    if train:
        out_file = r'{}database/train{}-seq{}.pickle'.format(data_output_path,cross_index,3)
        valid_file = r'{}database/test{}-seq{}.pickle'.format(data_output_path,cross_index,3)
    else:
        out_file = r'{}database/test{}-seq{}.pickle'.format(data_output_path,cross_index,3)

inputs = []
outputs = []
weights = None
num_seq = 3

for i in range(len(multi_opt_size)):
    opt = multi_opt_size[i]
    if opt == 0:
        input_x = Input(shape=(num_seq,224,224,3))
        inputs.append(input_x)

        x = keras.applications.mobilenet.MobileNet(
            input_shape=(224,224,3),
            pooling='avg',
            include_top=False,
            weights=weights,
            dropout=0.5
        )
        _x = TimeDistributed(x)(input_x)
        _x = LSTM(n_neurons, return_sequences=True)(_x)
        outputs.append(_x)
    else:
        # Temporal
        input_y = Input(shape=(num_seq,224,224,20))
        inputs.append(input_y)
        y = mobilenet.mobilenet_remake(
            name='temporal'+str(opt),
            input_shape=(224,224,depth),
            classes=classes,
            weight=weights
        )
        _y = TimeDistributed(y)(input_y)
        _y = LSTM(n_neurons, return_sequences=True)(_y)
        outputs.append(_y)

# Fusion
if fusion == 'avg':
    z = Average()(outputs)
elif fusion == 'max':
    z = Maximum()(outputs)
elif fusion == 'concat':
    z = Concatenate()(outputs)
elif fusion == 'conv':
    z = Concatenate()(outputs)
    z = Conv2D(256, (1, 1))(z)
else:
    z = Multiply()(outputs)

z = LSTM(n_neurons)(z)
z = Dropout(0.25)(z)
z = Dense(classes, activation='softmax')(z)

# Final touch
old_model = Model(inputs=inputs, outputs=z)

if train & (not retrain):
    old_model.load_weights('weights/multiple_lstm{}_{}_{}e_cr{}.h5'.format(opt_size,fusion,old_epochs,cross_index))

for layer in old_model.layers:
    layer.trainable = False

# New model
for i in range(len(outputs)):
    outputs[i] = TimeDistributed(Reshape((n_neurons,1)))(outputs[i])
t = Concatenate(axis=3)(outputs)
t = TimeDistributed(Conv1D(filters=1, kernel_size=1, use_bias=False))(t)
t = TimeDistributed(Flatten())(t)
t = LSTM(128)(t)
t = Dropout(0.25)(t)
t = Dense(classes, activation='softmax')(t)

result_model = Model(inputs=inputs, outputs=t)

# Run
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/multiple_lstm_weighted{}_{}_{}e_cr{}.h5'.format(opt_size,fusion,old_epochs,cross_index))

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)

    with open(valid_file,'rb') as f2:
        keys_valid = pickle.load(f2)
    len_valid = len(keys_valid)

    print('-'*40)
    print 'MobileNet Multiple {} stream: Training'.format(opt_size)
    print 'Number samples: {}'.format(len_samples)
    print 'Number valid: {}'.format(len_valid)
    print('-'*40)

    histories = []
    if server:
        steps = len_samples/batch_size
        validation_steps = int(np.ceil(len_valid*1.0/batch_size))
    else:
        steps = len_samples/batch_size
        validation_steps = int(np.ceil(len_valid*1.0/batch_size))
    
    for e in range(epochs):
        print('Epoch', e+1)
        print('-'*40)

        if server:
            random.shuffle(keys)

        time_start = time.time()

        history = result_model.fit_generator(
            gd.getTrainData(
                keys=keys,batch_size=batch_size,classes=classes,mode=3,train='train',opt_size=multi_opt_size,seq=True), 
            verbose=1, 
            max_queue_size=3, 
            steps_per_epoch=steps, 
            epochs=1,
            validation_data=gd.getTrainData(
                keys=keys_valid,batch_size=batch_size,classes=classes,mode=3,train='test',opt_size=multi_opt_size,seq=True),
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
        result_model.save_weights('weights/multiple_lstm_weighted{}_{}_{}e_cr{}.h5'.format(opt_size,fusion,old_epochs+1+e,cross_index))

        with open('histories/multiple_lstm_weighted{}_{}_{}_{}e_cr{}'.format(opt_size,fusion,old_epochs, epochs,cross_index), 'wb') as file_pi:
            pickle.dump(histories, file_pi)

else:
    result_model.load_weights('weights/multiple_lstm_weighted{}_{}_{}e_cr{}.h5'.format(opt_size,fusion,epochs,cross_index))

    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)

    len_samples = len(keys)

    print('-'*40)
    print 'MobileNet Multiple {} stream: Testing'.format(opt_size)
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
            keys=keys,batch_size=batch_size,classes=classes,mode=3,train='test',opt_size=multi_opt_size,seq=True), 
        max_queue_size=3, 
        steps=steps)

    run_time = time.time() - time_start
    
    y_classes = y_pred.argmax(axis=-1)
    print 'Score per samples'
    print(classification_report(Y_test, y_classes, digits=6))
    with open('results/multiple-lstm-weighted{}-{}-cr{}.txt'.format(opt_size,fusion,cross_index), 'w+') as fw1:
        fw1.write(classification_report(Y_test, y_classes, digits=6))
        fw1.write('\nRun time: ' + str(run_time))

    print 'Confusion matrix'
    print(confusion_matrix(Y_test, y_classes))
    with open('results/multiple-lstm-weighted{}-{}-cf-cr{}.txt'.format(opt_size,fusion,cross_index),'wb') as fw3:
        pickle.dump(confusion_matrix(Y_test, y_classes),fw3)

    print 'Run time: {}'.format(run_time)
