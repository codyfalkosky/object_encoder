import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, BatchNormalization, ReLU, Concatenate, Flatten, LayerNormalization
from tensorflow.keras.utils import plot_model


class ObjEncoder:
    '''
    Example:
        >>> model = ObjectEncoder().model
    '''

    def __init__(self, omit_structure=[], beta_clips=1, beta_coords=1, beta_combined=1, beta_em=1):
        '''
        Returns Object Encoder model at self.model
        '''
        self.model = self._build_model(omit_structure, beta_clips, beta_coords, beta_combined, beta_em)

    def _build_model(self, omit_structure, beta_clips, beta_coords, beta_combined, beta_em):
        '''
        Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
        and returning a vector of depth 256 encoding the image
        '''

        clips_in = Input([32, 40, 3], name='clips_input')

        ######## CLIP STAGE ########
        x = clips_in
        
        # Block 1
        if 'clip_block_1' not in omit_structure:
            x = Conv2D(64/beta_clips, (3, 3), padding='same', name='clip_block1_1')(x)
            x = LayerNormalization(                name='clip_block1_2')(x)
            x = ReLU(                              name='clip_block1_3')(x)
    
            x = Conv2D(64/beta_clips, (3, 3), padding='same', name='clip_block1_4')(x)
            x = LayerNormalization(                name='clip_block1_5')(x)
            x = ReLU(                              name='clip_block1_6')(x)

            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)

        # Block 2
        if 'clip_block_2' not in omit_structure:
            x = Conv2D(128/beta_clips,(3, 3), padding='same', name='clip_block2_1')(x)
            x = LayerNormalization(                name='clip_block2_2')(x)
            x = ReLU(                              name='clip_block2_3')(x)
    
            x = Conv2D(128/beta_clips,(3, 3), padding='same', name='clip_block2_4')(x)
            x = LayerNormalization(                name='clip_block2_5')(x)
            x = ReLU(                              name='clip_block2_6')(x)
    
            x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)

        # Block 3
        if 'clip_block_3' not in omit_structure:
            x = Conv2D(256/beta_clips,(3, 3), padding='same', name='clip_block3_1')(x)
            x = LayerNormalization(                name='clip_block3_2')(x)
            x = ReLU(                              name='clip_block3_3')(x)
    
            x = Conv2D(256/beta_clips,(3, 3), padding='same', name='clip_block3_4')(x)
            x = LayerNormalization(                name='clip_block3_5')(x)
            x = ReLU(                              name='clip_block3_6')(x)
    
            x = MaxPool2D((2, 2), 2,               name='clip_block3_7')(x)
        
        clips_out = Flatten()(x)

        ######## COORDS STAGE ########

        coords_in = Input([4], name='coords_input')

        x = coords_in
        # Block 1
        if 'coords_block_1' not in omit_structure:
            x = Dense(32/beta_coords,          name='coords_block1_1')(x)
            x = LayerNormalization(name='coords_block1_2')(x)
            x = ReLU(              name='coords_block1_3')(x)

        # Block 2
        if 'coords_block_2' not in omit_structure:
            x = Dense(64/beta_coords,          name='coords_block2_4')(x)
            x = LayerNormalization(name='coords_block2_5')(x)
            x = ReLU(              name='coords_block2_6')(x)

        coords_out = x

        ######## COMBINED STAGE ########
        
        x = Concatenate(       name='concatenate')([coords_out, clips_out])

        # Block 1
        if 'combined_block_1' not in omit_structure:
            x = Dense(1024/beta_combined,        name='combined_block1_1')(x)
            x = LayerNormalization(name='combined_block1_2')(x)
            x = ReLU(              name='combined_block1_3')(x)

        # Block 2
        if 'combined_block_2' not in omit_structure:
            x = Dense(512/beta_combined,         name='combined_block2_4')(x)
            x = LayerNormalization(name='combined_block2_5')(x)
            x = ReLU(              name='combined_block2_6')(x)

        x = Dense(256/beta_em,         name='output')(x)        

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
