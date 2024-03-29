import tensorflow as tf
from .data import Data
from .loss import LossC, LossE
from .model import ObjectEncoder
from .training import Training


class ObjectEncoder:
    def __init__(self, train_data_paths, valid_data_paths, loss='cosine'):
        
        # two alternate styles of calculating loss are possible
        self.loss_style = loss
        if loss == 'cosine':
            self.loss     = LossC().calc_loss
        elif loss == 'euclidian':
            self.loss     = LossE().calc_loss

        # load all datasets
        self.dataset  = {}
        self.dataset['train']  = Data(train_data_paths).dataset
        self.dataset['valid']  = Data(valid_data_paths).dataset

        # initalize model
        self.model    = ObjectEncoder().model

        # load training loops
        self.training = Training(self)

    def fit(self, optimizer, save_best_folder, save_below):
        self.training.fit(optimizer, save_best_folder, save_below)

# +
# obj_encoder = ObjEncoder(records_list = ['/Users/codyfalkosky/Desktop/Object_Encoder_Training/examples_3240.tfr'], 
#                          n_train_examples = 38)

# obj_encoder.fit(tf.keras.optimizers.Adam())
