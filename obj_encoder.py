import tensorflow as tf
from .data import Data
from .loss import LossC, LossE
from .model import ObjectEncoder
from .training import Training


class ObjEncoder:
    def __init__(self, records_list, n_train_examples, loss='cosine'):
        if loss == 'cosine':
            self.loss     = LossC().calc_loss
        elif loss == 'euclidian':
            self.loss     = LossE().calc_loss
                   
        self.dataset  = Data(records_list, n_train_examples).dataset
        self.model    = ObjectEncoder().model
        self.training = Training(self)

    def fit(self, optimizer):
        self.training.fit(optimizer)

# +
# obj_encoder = ObjEncoder(records_list = ['/Users/codyfalkosky/Desktop/Object_Encoder_Training/examples_3240.tfr'], 
#                          n_train_examples = 38)

# obj_encoder.fit(tf.keras.optimizers.Adam())
