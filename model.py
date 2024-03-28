import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, BatchNormalization, ReLU, Concatenate, Flatten
from tensorflow.keras.utils import plot_model


class ObjectEncoder:
    '''
    Example:
        >>> model = ObjectEncoder().model
    '''

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        '''
        Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,)]
        and returning a vector of depth 256 encoding the image
        '''

        clips_in = Input([32, 40, 3], name='clips_input')

        ######## CLIP STAGE ########

        # Block 1
        x = Conv2D(64, (3, 3), padding='same', name='clip_block1_1')(clips_in)
        x = BatchNormalization(                name='clip_block1_2')(x)
        x = ReLU(                              name='clip_block1_3')(x)

        x = Conv2D(64, (3, 3), padding='same', name='clip_block1_4')(x)
        x = BatchNormalization(                name='clip_block1_5')(x)
        x = ReLU(                              name='clip_block1_6')(x)

        x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)

        # Block 2
        x = Conv2D(128,(3, 3), padding='same', name='clip_block2_1')(x)
        x = BatchNormalization(                name='clip_block2_2')(x)
        x = ReLU(                              name='clip_block2_3')(x)

        x = Conv2D(128,(3, 3), padding='same', name='clip_block2_4')(x)
        x = BatchNormalization(                name='clip_block2_5')(x)
        x = ReLU(                              name='clip_block2_6')(x)

        x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)

        # Block 3
        x = Conv2D(256,(3, 3), padding='same', name='clip_block3_1')(x)
        x = BatchNormalization(                name='clip_block3_2')(x)
        x = ReLU(                              name='clip_block3_3')(x)

        x = Conv2D(256,(3, 3), padding='same', name='clip_block3_4')(x)
        x = BatchNormalization(                name='clip_block3_5')(x)
        x = ReLU(                              name='clip_block3_6')(x)

        x = MaxPool2D((2, 2), 2,               name='clip_block3_7')(x)
        clips_out = Flatten()(x)

        ######## COORDS STAGE ########

        coords_in = Input([4], name='coords_input')

        # Block 1
        x = Dense(32,          name='coords_block1_1')(coords_in)
        x = BatchNormalization(name='coords_block1_2')(x)
        x = ReLU(              name='coords_block1_3')(x)

        # Block 2
        x = Dense(64,          name='coords_block2_4')(x)
        x = BatchNormalization(name='coords_block2_5')(x)
        coords_out = ReLU(     name='coords_block2_6')(x)

        ######## COMBINED STAGE ########
        
        x = Concatenate(       name='concatenate')([coords_out, clips_out])

        # Block 1
        x = Dense(1024,        name='combined_block1_1')(x)
        x = BatchNormalization(name='combined_block1_2')(x)
        x = ReLU(              name='combined_block1_3')(x)

        # Block 2
        x = Dense(512,         name='combined_block2_4')(x)
        x = BatchNormalization(name='combined_block2_5')(x)
        x = ReLU(              name='combined_block2_6')(x)

        x = Dense(256,         name='output')(x)        

        model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')

        return model



if __name__ == '__main__':
    rows = 305
    clips_data    = tf.random.normal([rows, 32, 40, 3])
    coords_data   = tf.random.normal([rows, 4])
    siamese_model = SiameseModel()
    model_output  = siamese_model.model([clips_data, coords_data])
    # print(model_output)
    # plot_model(siamese_model.model, show_shapes=True, show_layer_names=True)
    siamese_model.model.summary()
