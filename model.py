import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, BatchNormalization, ReLU, Concatenate, Flatten, LayerNormalization, Dropout
from tensorflow.keras.utils import plot_model


class ObjEncoder:
    '''
    Example:
        >>> model = ObjectEncoder().model
    '''

    def __init__(self, architecture='MODEL1', model_params={}):
        '''
        Returns Object Encoder model at self.model
        '''
        self.model = self._build_model(architecture, model_params)

    def _build_model(self, architecture, model_params):
        if architecture == 'MODEL1':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(64, (3, 3), padding='same', name='clip_block1_1')(x)
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
    
            x = coords_in
            # Block 1
            x = Dense(32,          name='coords_block1_1')(x)
            x = BatchNormalization(name='coords_block1_2')(x)
            x = ReLU(              name='coords_block1_3')(x)
    
            # Block 2
            x = Dense(64,          name='coords_block2_4')(x)
            x = BatchNormalization(name='coords_block2_5')(x)
            x = ReLU(              name='coords_block2_6')(x)
    
            coords_out = x
    
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



        ################
        #### SIMPLE ####
        ################
        
        elif architecture == 'SIMPLE':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(16, (3, 3), padding='same', name='clip_block1_1')(x)
            x = ReLU(                              name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)
    
            # Block 2
            x = Conv2D(32, (3, 3), padding='same', name='clip_block2_1')(x)
            x = ReLU(                              name='clip_block2_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # Block 1
            x = Dense(16,          name='coords_block1_1')(x)
            x = ReLU(              name='coords_block1_3')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(32,          name='combined_block1_1')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(16,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

        #####################
        #### SIMPLE_DROP ####
        #####################
        
        elif architecture == 'SIMPLE_DROP':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(16, (3, 3), padding='same',     name='clip_block1_1')(x)
            x = ReLU(                                  name='clip_block1_3')(x)    
            x = Dropout(model_params['dropout_rate'],  name='clip_block1_d')(x)
            x = MaxPool2D((2, 2), 2,                   name='clip_block1_7')(x)
            
    
            # Block 2
            x = Conv2D(32, (3, 3), padding='same',     name='clip_block2_1')(x)
            x = ReLU(                                  name='clip_block2_3')(x)
            x = Dropout(model_params['dropout_rate'],  name='clip_block2_d')(x)
            x = MaxPool2D((2, 2), 2,                   name='clip_block2_7')(x)


            x = MaxPool2D((2, 2), 2,                   name='clip_block3_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # Block 1
            x = Dense(16,          name='coords_block1_1')(x)
            x = ReLU(              name='coords_block1_3')(x)
            x = Dropout(model_params['dropout_rate'],      name='coords_block1_d')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(32,                       name='combined_block1_1')(x)
            x = ReLU(                           name='combined_block1_3')(x)
            x = Dropout(model_params['dropout_rate'],   name='combined_block1_d')(x)

    
            x = Dense(16,                       name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

        #################
        #### SIMPLER ####
        #################
        
        elif architecture == 'SIMPLER_8':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(8, (3, 3), padding='same', name='clip_block1_1')(x)
            x = ReLU(                              name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)
    
            # Block 2
            x = Conv2D(16, (3, 3), padding='same', name='clip_block2_1')(x)
            x = ReLU(                              name='clip_block2_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # Block 1
            x = Dense(8,          name='coords_block1_1')(x)
            x = ReLU(              name='coords_block1_3')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(16,          name='combined_block1_1')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(8,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

        ###################
        #### CONV ONlY ####
        ###################
        
        elif architecture == 'CONV_ONLY':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(8, (8, 10), padding='same', name='clip_block1_1')(x)
            x = ReLU(                               name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,                name='clip_block1_7')(x)
    
            # Block 2
            x = Conv2D(16, (8, 10), padding='same', name='clip_block2_1')(x)
            x = ReLU(                               name='clip_block2_3')(x)    
            x = MaxPool2D((2, 2), 2,                name='clip_block2_7')(x)

            # Block 2
            x = Conv2D(32, (8, 10), padding='same', name='clip_block3_1')(x)
            x = ReLU(                               name='clip_block3_3')(x)    
            x = MaxPool2D((2, 2), 2,                name='clip_block3_7')(x)
            
            x = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            ######## COMBINED STAGE ########
    
            # Block 1
            x = Dense(16,          name='combined_block1_1')(x)
            x = BatchNormalization(name='combined_block1_2')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(8,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

        ###################
        #### SIMPLER 4 ####
        ###################
        
        elif architecture == 'SIMPLER_4':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(16, (3, 3), padding='same', name='clip_block1_1')(x)
            x = ReLU(                              name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)
    
            # Block 2
            x = Conv2D(32, (3, 3), padding='same', name='clip_block2_1')(x)
            x = ReLU(                              name='clip_block2_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)
  
            # Block 3
            x = Conv2D(64, (3, 3), padding='same', name='clip_block3_1')(x)
            x = ReLU(                              name='clip_block3_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block3_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # Block 1
            x = Dense(4,          name='coords_block1_1')(x)
            x = ReLU(              name='coords_block1_3')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(8,          name='combined_block1_1')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(4,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

        #################
        #### RC1 E4  ####
        #################
        
        elif architecture == 'RC1_E4':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(4, (3, 3), padding='same',  name='clip_block1_1')(x)
            x = ReLU(                              name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)
    
            # Block 2
            x = Conv2D(8, (3, 3), padding='same', name='clip_block2_1')(x)
            x = ReLU(                              name='clip_block2_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)
  
            # Block 3
            x = Conv2D(16, (3, 3), padding='same', name='clip_block3_1')(x)
            x = ReLU(                              name='clip_block3_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block3_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # Block 1
            x = Dense(4,          name='coords_block1_1')(x)
            x = ReLU(              name='coords_block1_3')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(8,          name='combined_block1_1')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(4,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

        #################
        #### BASIC 1 ####
        #################
        
        elif architecture == 'BASIC_1':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 8 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(8, (3, 3), padding='same',  name='clip_block1_1')(x)
            x = ReLU(                              name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # # Block 1
            # x = Dense(4,          name='coords_block1_1')(x)
            # x = ReLU(              name='coords_block1_3')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(8,          name='combined_block1_1')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(8,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

        #################
        #### BASIC 2 ####
        #################
        
        elif architecture == 'BASIC_2':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(8, (3, 3), padding='same',  name='clip_block1_1')(x)
            x = ReLU(                              name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)
    
            # # Block 2
            # x = Conv2D(16, (3, 3), padding='same', name='clip_block2_1')(x)
            # x = ReLU(                              name='clip_block2_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)
  
            # # Block 3
            # x = Conv2D(16, (3, 3), padding='same', name='clip_block3_1')(x)
            # x = ReLU(                              name='clip_block3_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block3_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # # Block 1
            # x = Dense(4,          name='coords_block1_1')(x)
            # x = ReLU(              name='coords_block1_3')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(8,          name='combined_block1_1')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(8,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model


        #################
        #### BASIC 3 ####
        #################
        
        elif architecture == 'BASIC_3':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(8, (3, 3), padding='same',  name='clip_block1_1')(x)
            x = ReLU(                              name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)
    
            # Block 2
            x = Conv2D(16, (3, 3), padding='same', name='clip_block2_1')(x)
            x = ReLU(                              name='clip_block2_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)
  
            # # Block 3
            # x = Conv2D(16, (3, 3), padding='same', name='clip_block3_1')(x)
            # x = ReLU(                              name='clip_block3_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block3_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # # Block 1
            # x = Dense(4,          name='coords_block1_1')(x)
            # x = ReLU(              name='coords_block1_3')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(8,          name='combined_block1_1')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(8,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

        #################
        #### BASIC 4 ####
        #################
        
        elif architecture == 'BASIC_4':
            '''
            Architecture for an object encoder taking an input of [image_tensor (32, 40, 3), coords_tensor (4,) cxcywh]
            and returning a vector of depth 256 encoding the image
            '''
    
            clips_in = Input([32, 40, 3], name='clips_input')
    
            ######## CLIP STAGE ########
            x = clips_in
            
            # Block 1   
            x = Conv2D(8, (3, 3), padding='same',  name='clip_block1_1')(x)
            x = ReLU(                              name='clip_block1_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block1_7')(x)
    
            # Block 2
            x = Conv2D(16, (3, 3), padding='same', name='clip_block2_1')(x)
            x = ReLU(                              name='clip_block2_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block2_7')(x)
  
            # # Block 3
            # x = Conv2D(16, (3, 3), padding='same', name='clip_block3_1')(x)
            # x = ReLU(                              name='clip_block3_3')(x)    
            x = MaxPool2D((2, 2), 2,               name='clip_block3_7')(x)
            
            clips_out = Flatten()(x)
    
            ######## COORDS STAGE ########
    
            coords_in = Input([4], name='coords_input')
    
            x = coords_in
            # # Block 1
            # x = Dense(4,          name='coords_block1_1')(x)
            # x = ReLU(              name='coords_block1_3')(x)
    
            coords_out = x
    
            ######## COMBINED STAGE ########
            
            x = Concatenate(       name='concatenate')([coords_out, clips_out])
    
            # Block 1
            x = Dense(16,          name='combined_block1_1')(x)
            x = ReLU(              name='combined_block1_3')(x)

    
            x = Dense(16,         name='output')(x)        
    
            model = tf.keras.Model(inputs=[clips_in, coords_in], outputs=[x], name='obj_encoder')
    
            return model

if __name__ == '__main__':
    obj_encoder = ObjEncoder('BASIC_2')
    obj_encoder.model.summary()
