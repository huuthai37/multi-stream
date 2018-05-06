from keras import backend as K

def consensus_categorical_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum(y_true * (y_pred - K.logsumexp(y_pred)), axis=-1)

def consensus_timeseries_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum(y_true * (y_pred + K.log(y_pred) - K.logsumexp(y_pred)), axis=-1) / 2

# y_pred = [0.1,0.2,0.3,0.4]
# y_true = [1.,0.,0.,0.]
# old = -K.sum(y_true * K.log(y_pred), axis=-1)
# print K.eval(old)
# g = y_pred - K.logsumexp(y_pred)
# loss = -K.sum(y_true*g, axis=-1)
# print K.eval(loss)
