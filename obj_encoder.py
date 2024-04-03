import tensorflow as tf
from .data import Data
from .loss import LossC, LossE
from .model import ObjEncoder
from .training import Training
from .decode import decode_to_labels


class ObjectEncoder:
    def __init__(self, train_data_paths=None, valid_data_paths=None, loss='cosine',
                load_weights=None, with_margin=False, verbose=0, architecture='MODEL1'):
        '''
        Top level class for building, training, loading and predicting object encodings

        Args:
            train_data_paths: list of str, or None
                if format ['path/to/data_train_1.tfr', 'path/to/data_train_2.tfr', ...]
                leave as None if only predicting with loaded weights
            valid_data_paths: list of str, or None
                if format ['path/to/data_valid_1.tfr', 'path/to/data_valid_2.tfr', ...]
                leave as None if only predicting with loaded weights
            loss: str 'cosine' or 'euclidian'
                a switch that controls which type of loss calculation is loaded
                cosine for basing loss off of encoded vector cosine similarity
                euclidian for basing loss off of encoded vector euclidian distance
            load_weights: None or str
                if None weights are initialized at random
                if path/to/model.weights.h5 weights are loaded
            with_margin: bool
                switch for loss calculation
                False uses default values to 'push' toward without ever reaching
                True uses a margin for which performance beyond is ignored
            verbose: int
                for debug, 0 to silence all messages
                
        '''
        
        # two alternate styles of calculating loss are possible
        self.loss_style = loss
        self.verbose = verbose
        if loss == 'cosine':
            self.loss_obj = LossC(self)

            if with_margin:
                self.loss     = self.loss_obj.calc_loss_with_margin
            else:
                self.loss     = self.loss_obj.calc_loss
        elif loss == 'euclidian':
            self.loss_obj     = LossE(self)
            self.loss         = self.loss_obj.calc_loss
            
        # load all datasets
        self.dataset  = {}
        if train_data_paths:
            self.dataset['train']  = Data(train_data_paths).dataset
        if valid_data_paths:
            self.dataset['valid']  = Data(valid_data_paths).dataset

        # initalize model
        self.model    = ObjEncoder(architecture).model

        if load_weights:
            self.model.load_weights(load_weights)
            if self.verbose > 0:
                print(f'loaded weights at {load_weights}')
            

        # load training loops
        self.training = Training(self)

    def fit(self, optimizer=None, save_best_folder=None, save_below=None, epochs=None, 
            similarity_threshold=.99, percentile=.3, decode_basis='cosine', euclidean_thresh=.5,
            sim_thresh=.95, dif_thresh=.5):
        '''
        top level fit function, to be called directally when training

        Args:
            optimizer: tf.keras.optimizers.Optimizer
                if None uses exsisting optimizer, for case when updating
                optimizer parametes mid training
            save_best_folder: string path/to/folder
                for saving models with performance better than save_below
                if None, no model saving
            save_below: float
                threshold for weight saving worthy performance
            epochs: int
                number of epochs to stop training
            similarity_threshold: float
                strickly for reporting accuracy, cosine similarity minimum to be considered for label clustering
            percentile: float
                percentile of cluster connection strength to be considered valid connection
                read more in function decode_to_labels in decode.py

        Returns:
            trained model at self.model            
        '''
        self.loss_obj.sim_thresh = sim_thresh
        self.loss_obj.dif_thresh = dif_thresh
        
        self.training.fit(optimizer, save_best_folder, save_below, epochs, similarity_threshold, percentile,
                         decode_basis, euclidean_thresh)

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
# -

tf.keras.optimizers.Optimizer


