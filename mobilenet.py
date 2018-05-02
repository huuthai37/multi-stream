import keras
import sys
import get_data as gd
from keras.models import Model
from keras import backend as K
from keras.layers import DepthwiseConv2D, Dense, Conv2D, Activation, BatchNormalization, Reshape, Flatten, Input, ZeroPadding2D, Average, GlobalAveragePooling2D, Concatenate

def relu6(x):
    return K.relu(x, max_value=6)

def mobilenet_remake(name, input_shape, classes, weight=None, non_train=False, depth=20):
    
    mobilenet = keras.applications.mobilenet.MobileNet(
        input_shape=(224,224,3),
        pooling='avg',
        include_top=False,
        weights=weight,
        dropout=0.5
    )
    name = name + '_'

    # Disassemble layers
    layers = [l for l in mobilenet.layers]

    new_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(1, 1))(new_input)
    x = Conv2D(filters=32, 
              kernel_size=(3, 3),
              padding='valid',
              use_bias=False,
              name=name+'conv_new',
              strides=(2,2))(x)

    for i in range(3, len(layers)):
        layers[i].name = str(name) + layers[i].name
        x = layers[i](x)

    model = Model(inputs=new_input, outputs=x)
    if weight is not None:
        model.get_layer(name+'conv_new').set_weights(gd.convert_weights(layers[2].get_weights(), depth))

    return model

def mobilenet_by_me(name, inputs, input_shape, classes, weight = '', cut = 5, non_train=False):
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,
        dropout=0.5
    )
    name = name + '_'

    # Disassemble layers
    layers = [l for l in model.layers]

    new_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(1, 1))(new_input)
    x = Conv2D(filters=32, 
              kernel_size=(3, 3),
              padding='valid',
              use_bias=False,
              strides=(2,2))(x)

    for i in range(3, len(layers)-3):
        layers[i].name = str(name) + layers[i].name
        x = layers[i](x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model2 = Model(inputs=new_input, outputs=x)
    if weight != '':
        model2.load_weights(weight)

    new_layers = [l for l in model2.layers]

    y = inputs
    for i in range(0, len(new_layers)-cut):
        if non_train:
            new_layers[i].trainable = False
        y = new_layers[i](y)

    return y

def mobilenet_new(name, inputs, input_shape, classes, weight = '', concat_input=False, cut=5):
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,
        dropout=0.5,
    )

    # Disassemble layers
    layers = [l for l in model.layers]
    name = name + '_'
    if concat_input:
        new_input = Concatenate()(inputs)
        x = ZeroPadding2D(padding=(1, 1))(new_input)
        x = Conv2D(filters=32, 
                  kernel_size=(3, 3),
                  padding='valid',
                  use_bias=False,
                  strides=(2,2),
                  name=name+'conv_first_new')(x)
    else:
        length = len(inputs)
        outs = inputs
        for i in range(length):
            outs[i] = ZeroPadding2D(padding=(1, 1))(outs[i])
            outs[i] = Conv2D(filters=32, 
                      kernel_size=(3, 3),
                      padding='valid',
                      use_bias=False,
                      strides=(2,2))(outs[i])
        x = Concatenate()(outs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(relu6)(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = DepthwiseConv2D((3, 3),
                            padding='valid',
                            depth_multiplier=1,
                            strides=(1, 1),
                            use_bias=False)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(relu6)(x)
        x = Conv2D(64, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name=name+'conv_last_new')(x)

    if concat_input:
        for i in range(3, len(layers)-3-cut):
            layers[i].name = str(name) + layers[i].name
            x = layers[i](x)
    else:
        for i in range(10, len(layers)-3-cut):
            layers[i].name = str(name) + layers[i].name
            x = layers[i](x)

    # x = Flatten()(x)
    # x = Dense(classes, activation='softmax')(x)

    return x
