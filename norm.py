from keras import backend as K
from keras.constraints import Constraint

class SumNorm(Constraint):

    def __call__(self, w):
        return K.abs(w) / (K.epsilon() + K.sum(K.abs(w)))