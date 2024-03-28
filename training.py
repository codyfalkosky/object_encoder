import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.notebook import tqdm
from IPython.display import clear_output


# +
class Training:
    '''
    All Training and Validation Functions for Object Encoder training
    '''
    def __init__(self, parent_obj):
        self.parent_obj   = parent_obj
        self.train_loss   = []
        self.valid_loss   = []
        
        self.train_metric = tf.keras.metrics.Mean()
        self.valid_metric = tf.keras.metrics.Mean()

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            model_out = self.parent_obj.model([batch['objects'], batch['coords']], training=True)
            loss      = self.parent_obj.loss(model_out, batch['labels'], self.train_metric)

        gradients = tape.gradient(loss, self.parent_obj.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.parent_obj.model.trainable_variables))

    def valid_step(self, batch):
        model_out = self.parent_obj.model([batch['objects'], batch['coords']], training=False)
        loss      = self.parent_obj.loss(model_out, batch['labels'], self.valid_metric)

    def plot_loss(self):
        '''
        for visualization during training
        displays train and valid loss
        '''
        clear_output(wait=True)

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
        plt.show()

    def fit(self, optimizer):
        '''
        basic fit function for object encoder

        Args:
            optimizer (keras.src.optimizers) : An optimizer instance

        Returns:
            trained model weights to model at self.parent_obj.model

        This function is meant to be called by the parent_obj's fit method
        '''
        self.optimizer = optimizer

        print('Loading Training Data')
        train_data_len = 0
        for batch in tqdm(self.parent_obj.dataset['train']):
            train_data_len += 1

        print('Loading Validation Data')
        valid_data_len = 0
        for batch in tqdm(self.parent_obj.dataset['valid']):
            valid_data_len += 1
        

        while True:
            print('Training Epoch:')
            for batch in tqdm(self.parent_obj.dataset['train'], total=train_data_len):
                self.train_step(batch)

            print('Validation Epoch:')
            for batch in tqdm(self.parent_obj.dataset['valid'], total=valid_data_len):
                self.valid_step(batch)

            # record and reset train metric
            self.train_loss.append(self.train_metric.result().numpy())
            self.train_metric.reset_states()

            # record and reset valid metric
            self.valid_loss.append(self.valid_metric.result().numpy())
            self.valid_metric.reset_states()

            # show loss
            self.plot_loss()
    
    
