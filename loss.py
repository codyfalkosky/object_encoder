import tensorflow as tf
from .decode import decode_to_labels


# +
# ########
# siamese_model = SiameseModel()
# clips_data    = tf.random.uniform([10, 32, 40, 3])
# coords_data   = tf.random.uniform([10, 4  ])
# labels_data   = tf.convert_to_tensor([1., 2., 3., 1., 2., 3., 1., 2., 3., 4.])
# model_out     = siamese_model.model([clips_data, coords_data])
# ########
# -

class LossC:
    'Siamese network loss using cosine similarity'

    def _cosine_similiarty(self, x):
        '''
        given a matrix a, computes cosine similarity of all vectors in a to all other vectors in a
        '''
        
        # normalize all vectors to magnitude 1
        norm       = tf.norm(x, axis=1, keepdims=True)
        x_n        = x / norm

        # calculate cosine similarity
        sim_matrix = tf.matmul(x_n, x_n, transpose_b=True)

        return sim_matrix

    def _label_mask(self, labels):
        '''
        generates mask for all unique objects

        Args:
            labels (tensor) : shape (n_obj,)
        Returns:
            same_obj (tensor) : bool shape [n_labels, is_label, matches]
                for comparing an object to all the other versions of itself
            diff_obj (tensor) : bool shape [n_labels, is_label, not_matches]
                for compaering an object to everything that it isn't
        '''
        y, idx   = tf.unique(labels)
        mask_2d  = y[:, None] == labels[None, :]             # [n_labels, is_label ]
        same_obj = mask_2d[:, :, None] & mask_2d[:, None, :] # [n_labels, is_label, matches]
        # diff_obj = mask_2d[:, :, None] & tf.logical_not(mask_2d)[:, None, :] # [n_labesl, is_label, not_matches]

        same_obj = tf.reduce_any(same_obj, axis=0)
        # diff_obj = tf.reduce_any(diff_obj, axis=0)
        return same_obj

    def _cosine_sim_true(self, labels):
        '''
        calculates perfect score similary matrix for loss
        
        Args:
            labels (tensor) : shape (n_obj,)
        Returns:
            y_true (tensor) : shape (n_obj, n_obj)
                same shape as the cosine similarity matrix 
        
        '''

        same_obj = self._label_mask(labels)

        y_true = tf.where(same_obj, 1, -1)
        return y_true

    def _resolve_labels(y_true, y_pred):
        y_pred    = np.array(y_pred)
        y_true    = np.array(y_true)
        unique    = np.unique(y_pred)
        labels    = np.zeros_like(y_pred)
    
        for u in unique:
            mask = y_pred == u
    
            mode = statistics.mode(y_true[mask])
            
            labels[mask] =  mode
    
        return labels
    
    def _get_accuracy(y_true_labels, y_pred_labels):
        y_pred_labels = _resolve_labels(y_true_labels, y_pred_labels)
        
        accuracy = (y_true_labels == y_pred_labels).mean()
    
        return accuracy
        
    def calc_loss(self, model_output, labels, loss_metric, accuracy_metric):
        '''
        Calculates loss for siamese network

        Args:
            model_output (tensor) : shape (n_obj, encoding_depth)
            labels       (tensor) : shape (n_obj,)

        Returns:
            loss (tensor) : single scalar loss value
        
        '''
        y_true = self._cosine_sim_true(labels)
        y_true = tf.reshape(y_true, [-1])
        
        y_pred = self._cosine_similiarty(model_output)
        y_pred = tf.reshape(y_pred, [-1])

        loss   = tf.keras.losses.mean_squared_error(y_true, y_pred)

        loss_metric.update_state(loss, sample_weight=y_pred.shape[0])

        ##### ACCURACY #####

        y_true_labels = labels
        y_pred_labels = decode_to_labels(model_output)
               
        accuracy = _get_accuracy(y_true_labels, y_pred_labels)        
        accuracy_metric.update_state(accuracy, sample_weight=y_pred.shape[0])
               
        return loss


class LossE:
    'Siamese network loss using euclidian distance'

    def _euclidean_distance(self, x):
        
        '||p - q||^2 = ||p||^2 + ||q||^2 - 2 * p dot q.T '
        
        # compute parts
        squared_norms = tf.reduce_sum(tf.square(x), axis=1)
    
        p_sn = tf.reshape(squared_norms, [-1, 1])
        q_sn = tf.reshape(squared_norms, [1 ,-1])
    
        # ||p||^2 + ||q||^2 - 2 * p * q.T
        squared_distance = (p_sn + q_sn) - 2 * tf.matmul(x, x, transpose_b=True)
    
        # clip for floating point stability
        squared_distance = tf.maximum(squared_distance, 0.)
    
        # square root for distance
        distance = tf.sqrt(squared_distance)
    
        return distance

    def _label_mask(self, labels):
        '''
        generates mask for all unique objects

        Args:
            labels (tensor) : shape (n_obj,)
        Returns:
            same_obj (tensor) : bool shape [n_labels, is_label, matches]
                for comparing an object to all the other versions of itself
            diff_obj (tensor) : bool shape [n_labels, is_label, not_matches]
                for compaering an object to everything that it isn't
        '''
        y, idx   = tf.unique(labels)
        mask_2d  = y[:, None] == labels[None, :]             # [n_labels, is_label ]
        same_obj = mask_2d[:, :, None] & mask_2d[:, None, :] # [n_labels, is_label, matches]
        # diff_obj = mask_2d[:, :, None] & tf.logical_not(mask_2d)[:, None, :] # [n_labesl, is_label, not_matches]

        same_obj = tf.reduce_any(same_obj, axis=0)
        # diff_obj = tf.reduce_any(diff_obj, axis=0)
        return same_obj

    def _euclidean_distance_true(self, labels):
        '''
        calculates perfect score similary matrix for loss
        
        Args:
            labels (tensor) : shape (n_obj,)
        Returns:
            y_true (tensor) : shape (n_obj, n_obj)
                same shape as the cosine similarity matrix 
        
        '''

        same_obj = self._label_mask(labels)

        y_true = tf.where(same_obj, 0, 10)
        return y_true
        
    def calc_loss(self, model_output, labels, loss_metric):
        '''
        Calculates loss for siamese network

        Args:
            model_output (tensor) : shape (n_obj, encoding_depth)
            labels       (tensor) : shape (n_obj,)

        Returns:
            loss (tensor) : single scalar loss value
        
        '''
        y_true = self._euclidean_distance_true(labels)
        y_true = tf.reshape(y_true, [-1])
        
        y_pred = self._euclidean_distance(model_output)
        y_pred = tf.reshape(y_pred, [-1])

        loss   = tf.keras.losses.mean_squared_error(y_true, y_pred)

        # loss_metric.update_state(loss, sample_weight=y_pred.shape[0])
        
        return loss
