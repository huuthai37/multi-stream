import sys
import pickle
import random
import numpy as np
import config
from sklearn.metrics import classification_report
import mobilenet
import keras
from keras import optimizers
import get_data as gd
from keras.models import Model
from keras.layers import Dense, Concatenate, GlobalAveragePooling2D, Dropout, Reshape, Flatten, Input

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

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

classes = 11
depth = 20
drop_rate = 0.5

model = keras.applications.mobilenet.MobileNet(
    include_top=True,
    dropout=0.5,
    weights='imagenet'
)

# Disassemble layers
layers = [l for l in model.layers]

# input_x = Input(shape=(224,224,3))

# x = mobilenet.mobilenet_by_me(
#     name='spatial', 
#     inputs=input_x, 
#     input_shape=(224,224,3), 
#     classes=classes)

# Temporal
input_y1 = Input(shape=(224,224,20))
input_y2 = Input(shape=(224,224,20))
# input_y3 = Input(shape=(224,224,20))
inputs = [input_y1,input_y2]

y = mobilenet.mobilenet_new(
    name='temporal', 
    inputs=inputs, 
    input_shape=(224,224,20), 
    classes=classes,
    cut=0
    )

# z = Concatenate()([x, y])
# z = GlobalAveragePooling2D()(z)
# z = Reshape((1,1,2048))(z)
# z = Dropout(0.5)(z)
# z = Flatten()(z)
# z = Dense(classes, activation='softmax')(z)

y = Flatten()(y)
y = Dense(classes, activation='softmax')(y)
# Final touch
result_model = Model(inputs=[input_y1,input_y2], outputs=y)
# result_model.summary()

length = len(layers)
k = 4
for i in range(4,5): # 6,9
    result_model.layers[i].set_weights(gd.convert_weights(layers[2].get_weights(), depth))
    print('Set weight', i)
for i in range(3, length-2):
    temp_weights = result_model.layers[i+k].get_weights()
    if temp_weights:
        if i > 9:
            result_model.layers[i+k].set_weights(layers[i].get_weights())
        if (i < 9) & (i != 6):
            result_model.layers[i+k].set_weights(gd.concat_weights(layers[i].get_weights(), len(inputs), len(temp_weights)))
        if i == 6:
            temp_weights_new = gd.convert_weights(layers[i+k].get_weights(), 96, 3, 1)
            result_model.layers[i+k].set_weights(gd.concat_weights(temp_weights_new, 1, len(temp_weights)))
        if i == 9:
            result_model.layers[i+k].set_weights(gd.convert_weights(layers[i].get_weights(), 96, 1, 64))
    print('Set weight', i+k)

result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

result_model.save_weights('weights/mobilenet_temporal_multi2_0e.h5')

# opt_size = 2
# batch_size = int(sys.argv[1])
# out_file = r'/home/oanhnt/thainh/data/database/train-opt{}.pickle'.format(opt_size)
# with open(out_file,'rb') as f1:
#     keys = pickle.load(f1)
# len_samples = len(keys)

# result_model.fit_generator(
#     gd.getTrainData(
#         keys,
#         batch_size,
#         classes,
#         4,
#         'train', 
#         opt_size), 
#     verbose=1, 
#     max_queue_size=2, 
#     steps_per_epoch=len_samples/batch_size, 
#     epochs=1,
# )


