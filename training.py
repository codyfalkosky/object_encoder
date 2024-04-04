import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.notebook import tqdm
from IPython.display import clear_output
import pickle


# +
class Training:
    '''
    All Training and Validation Functions for Object Encoder training
    '''
    def __init__(self, parent_obj):
        self.parent_obj   = parent_obj
        self.train_loss   = []
        self.valid_loss   = []
        self.train_accuracy = []
        self.valid_accuracy = []
        
        self.train_metric = tf.keras.metrics.Mean()
        self.valid_metric = tf.keras.metrics.Mean()

        self.train_accuracy_metric = tf.keras.metrics.Mean()
        self.valid_accuracy_metric = tf.keras.metrics.Mean()

    def train_step(self, batch, decode_params):
        '''
        basic training step

        Args:
            batch: iterable of tensors (objects, coords)
                objects: tensor
                    shape [b, 32, 40, 3] images of cars
                coords: tensor
                    shape [b, 4] cxcywh coords
            decode_params: dictionary containing
                similarity_threshold: float
                    strickly for reporting accuracy, cosine similarity minimum to be considered for label clustering
                percentile: float
                    percentile of cluster connection strength to be considered valid connection
                    read more in function decode_to_labels in decode.py

        Returns:
            weight updates to model at self.parent_obj.model
            update to metric for loss 
            update to metric to accuracy 
        '''
        with tf.GradientTape() as tape:
            model_out = self.parent_obj.model([batch['objects'], batch['coords']], training=True)
            loss      = self.parent_obj.loss(model_out, batch['labels'], self.train_metric, self.train_accuracy_metric,
                                             decode_params, for_='train')

        if not tf.math.is_nan(loss):
            gradients = tape.gradient(loss, self.parent_obj.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.parent_obj.model.trainable_variables))

    def valid_step(self, batch, decode_params):
        '''
        basic valid step

        Args:
            batch: iterable of tensors (objects, coords)
                objects: tensor
                    shape [b, 32, 40, 3] images of cars
                coords: tensor
                    shape [b, 4] cxcywh coords
            decode_params: dictionary containing
                similarity_threshold: float
                    strickly for reporting accuracy, cosine similarity minimum to be considered for label clustering
                percentile: float
                    percentile of cluster connection strength to be considered valid connection
                    read more in function decode_to_labels in decode.py

        Returns:
            weight updates to model at self.parent_obj.model
            update to metric for loss 
            update to metric to accuracy 
        '''
        model_out = self.parent_obj.model([batch['objects'], batch['coords']], training=False)
        loss      = self.parent_obj.loss(model_out, batch['labels'], self.valid_metric, self.valid_accuracy_metric,
                                             decode_params, for_='valid')

    def plot_loss(self):
        '''
        for visualization during training
        displays train and valid loss, and accuracy
        '''
        clear_output(wait=True)

        plt.figure(figsize=(15, 5))
        plt.subplot(1,3,1)
        plt.title(f"Last Epoch Valid Loss: {self.valid_loss[-1]:.5f}")
        plt.plot(self.train_loss,  color='C0')
        plt.plot(self.valid_loss,  color='C1')

        min_idx = np.array(self.valid_loss).argmin()
        min_val = np.array(self.valid_loss).min()
        
        plt.scatter(min_idx, min_val, marker='x', color='C3')
        plt.text(min_idx, min_val, round(min_val, 4), fontsize='x-small', ha='left', va='top')
        try:
            plt.ylim([0, self.valid_loss[-1]*3])
        except:
            plt.ylim([0, 1])


        plt.subplot(1,3,2)
        plt.title('Accuracy')

        plt.plot(self.train_accuracy, color='C0')
        plt.plot(self.valid_accuracy, color='C1')
        plt.ylim([0, 1])

        plt.subplot(1,3,3)

        t_len = len(self.parent_obj.loss_obj.train_cos_sim)
        v_len = len(self.parent_obj.loss_obj.valid_cos_sim)

        plt.title('Embedding Vectors: cos_sim')
        color_dict = {0:'C0', 1:'C1'}
        c = [color_dict[n] for n in self.parent_obj.loss_obj.train_labels]
        plt.scatter(range(t_len), self.parent_obj.loss_obj.train_cos_sim, color=c, marker='.', alpha=.2, s=1)
        color_dict = {0:'C2', 1:'C3'}
        c = [color_dict[n] for n in self.parent_obj.loss_obj.valid_labels]
        plt.scatter(range(t_len, t_len+v_len), self.parent_obj.loss_obj.valid_cos_sim, color=c, marker='.', 
                    alpha=.2, s=1)
        plt.ylim([-1, 1])
        plt.xticks([])
        if self.save_fig_folder:
            plt.savefig(f'{self.save_fig_folder}/fig_{len(self.train_accuracy)}.png')
        plt.show()

    def save_best(self, save_best_folder, save_below):
        '''
        saves best model to save_best_folder, if save_best_folder = '' does nothing

        Args:
            save_best_folder (str) : like "path/to/save/folder" # no final forward-slash
        Returns:
            model saved to save_best_folder/yolov2_model_{str_loss}.h5"
        '''


        if self.valid_loss[-1] == min(self.valid_loss):
            if self.valid_loss[-1] < save_below:
                str_loss = f"{self.valid_loss[-1]:.5f}"
                str_loss = str_loss.replace('.', '')
                self.parent_obj.model.save_weights(f"{save_best_folder}/obj_encoder_model_{self.parent_obj.loss_style}_{str_loss}.weights.h5")

    def save_history(self, run, save_dir):
        '''
        saves training history to save_dir

        run: int or string
            number to represent training run i.e. run 16
        save_dir: string path/to/folder
            history will be saved to this folder
        '''
        history = {'train loss'    : self.train_loss,
                   'valid loss'    : self.valid_loss,
                   'train accuracy': self.train_accuracy,
                   'valid accuracy': self.valid_accuracy,
                   'optimizer'     : self.optimizer.get_config(),
                   'loss_style'    : self.parent_obj.loss_style}

        with open(f'{save_dir}/run_{run}_history.pkl', 'wb') as file:
            pickle.dump(history, file)


    def fit(self, optimizer, save_best_folder, save_below, epochs, similarity_threshold, percentile,
           decode_basis, euclidean_thresh):
        '''
        basic fit function for object encoder

        Args:
            optimizer (keras.src.optimizers) : An optimizer instance

        Returns:
            trained model weights to model at self.parent_obj.model

        This function is meant to be called by the parent_obj's fit method
        '''
        if optimizer:
            self.optimizer = optimizer

        decode_params = {'similarity_threshold' : similarity_threshold, 'percentile': percentile,
                         'decode_basis': decode_basis, 'euclidean_thresh':euclidean_thresh}

        print('Loading Training Data')
        train_data_len = 0
        for batch in tqdm(self.parent_obj.dataset['train']):
            train_data_len += 1

        print('Loading Validation Data')
        valid_data_len = 0
        for batch in tqdm(self.parent_obj.dataset['valid']):
            valid_data_len += 1
        

        while True:

            self.parent_obj.loss_obj.train_cos_sim = []
            self.parent_obj.loss_obj.valid_cos_sim = []
            self.parent_obj.loss_obj.train_labels  = []
            self.parent_obj.loss_obj.valid_labels  = []
        
            # print(f'Training Epoch: {len(self.train_loss)}')
            print(f'{len(self.train_loss)}', end=' ')
            # for batch in tqdm(self.parent_obj.dataset['train'], total=train_data_len):
            for batch in self.parent_obj.dataset['train']:
                self.train_step(batch, decode_params)

            # print('Validation Epoch:')
            # for batch in tqdm(self.parent_obj.dataset['valid'], total=valid_data_len):
            for batch in self.parent_obj.dataset['valid']:
                self.valid_step(batch, decode_params)

            # record and reset train loss metric
            self.train_loss.append(self.train_metric.result().numpy())
            self.train_metric.reset_state()

            # record and reset train accuracy metric
            self.train_accuracy.append(self.train_accuracy_metric.result().numpy())
            self.train_accuracy_metric.reset_state()

            # record and reset valid loss metric
            self.valid_loss.append(self.valid_metric.result().numpy())
            self.valid_metric.reset_state()

            # record and reset valid accuracy metric
            self.valid_accuracy.append(self.valid_accuracy_metric.result().numpy())
            self.valid_accuracy_metric.reset_state()

            # save if best
            if save_best_folder:
                self.save_best(save_best_folder, save_below)

            # show loss
            if len(self.train_loss) % self.save_fig_every == 0:
                self.plot_loss()


            if epochs:
                if len(self.train_accuracy) > epochs:
                    break
    
    
