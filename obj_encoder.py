import tensorflow as tf
from .data import Data
from .loss import LossC, LossE
from .model import ObjEncoder
from .training import Training
from .decode import decode_to_labels


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
        self.model    = ObjEncoder().model

        # load training loops
        self.training = Training(self)

    def fit(self, optimizer, save_best_folder, save_below):
        self.training.fit(optimizer, save_best_folder, save_below)

    def label(self, model_inputs, **kwargs):
        model_output = self.model(model_inputs, training=False)
        labels       = decode_to_labels(model_output, **kwargs)
        return labels

# +
# obj_encoder = ObjEncoder(records_list = ['/Users/codyfalkosky/Desktop/Object_Encoder_Training/examples_3240.tfr'], 
#                          n_train_examples = 38)

# obj_encoder.fit(tf.keras.optimizers.Adam())
