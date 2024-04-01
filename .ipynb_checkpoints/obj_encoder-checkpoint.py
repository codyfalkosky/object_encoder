import tensorflow as tf
from .data import Data
from .loss import LossC, LossE
from .model import ObjEncoder
from .training import Training
from .decode import decode_to_labels


class ObjectEncoder:
    def __init__(self, train_data_paths=None, valid_data_paths=None, loss='cosine',
                load_weights=None, verbose=0):
        
        # two alternate styles of calculating loss are possible
        self.loss_style = loss
        if loss == 'cosine':
            self.loss_obj = LossC()
            self.loss     = self.loss_obj.calc_loss
        elif loss == 'euclidian':
            self.loss_obj     = LossE()
            self.loss         = self.loss_obj.calc_loss
            
        # load all datasets
        self.dataset  = {}
        if train_data_paths:
            self.dataset['train']  = Data(train_data_paths).dataset
        if valid_data_paths:
            self.dataset['valid']  = Data(valid_data_paths).dataset

        # initalize model
        self.model    = ObjEncoder().model

        if load_weights:
            self.model.load_weights(load_weights)
            if verbose > 0:
                print(f'loaded weights at {load_weights}')
            

        # load training loops
        self.training = Training(self)

    def fit(self, optimizer, save_best_folder, save_below, epochs=None, similarity_threshold=.99, percentile=.3):
        self.training.fit(optimizer, save_best_folder, save_below, epochs, similarity_threshold, percentile)

    def label(self, model_inputs, **kwargs):
        '''
        decodes object_encoder model_output to labels
    
        Args:
            model_inputs (list) : [objects, boxes]
                objects (tensor) : shape [n_obj, 32, 40, 3]
                boxes (tensor)   : shape [n_obj, 4]

            **kwargs: hyperparameters for decode_to_labels
                similarity_threshold (float) : minimum cosine similary threshold for considering two encodings similar
                percentile (float)           : percentile of cluster connection strength to consider a connection valid
        Return:
            labels (list) : every vector labeled by cluster
        '''
        model_output = self.model(model_inputs, training=False)
        labels       = decode_to_labels(model_output, **kwargs)
        return labels

# +
# obj_encoder = ObjEncoder(records_list = ['/Users/codyfalkosky/Desktop/Object_Encoder_Training/examples_3240.tfr'], 
#                          n_train_examples = 38)

# obj_encoder.fit(tf.keras.optimizers.Adam())
