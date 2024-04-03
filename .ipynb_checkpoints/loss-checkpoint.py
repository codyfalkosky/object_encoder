import tensorflow as tf
import numpy as np
import statistics
from .decode import decode_to_labels

# ########
# siamese_model = SiameseModel()
# clips_data    = tf.random.uniform([10, 32, 40, 3])
# coords_data   = tf.random.uniform([10, 4  ])
labels   = tf.convert_to_tensor([1., 2., 3., 1., 2., 3., 1., 2., 3., 4.])
# model_out     = siamese_model.model([clips_data, coords_data])
# ########

class LossC:
    'Object Encoder loss using cosine similarity'
    def __init__(self, parent_obj):
        self.parent_obj = parent_obj

    def _cosine_similiarty(self, x):
        '''
        given a matrix a, computes cosine similarity of all vectors in a to all other vectors in a

        Args:
            x: tensor [b, encodings]
                the output of the object encoding model
        Returns:
            sim_matrix: tensor [b, b]
                cosine_similarity of all vectors to all other vectors
        
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
            labels: tensor
                shape (n_obj,)
        Returns:
            same_obj: tensor
                bool shape [n_labels, is_label, matches]
                for comparing an object to all the other versions of itself
            diff_obj: tensor
                bool shape [n_labels, is_label, not_matches]
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
            labels: tensor
                shape (n_obj,)
        Returns:
            y_true: tensor
                shape (n_obj, n_obj)
                same shape as the cosine similarity matrix 
        
        '''

        same_obj = self._label_mask(labels)

        y_true = tf.where(same_obj, 1, -1)
        return y_true

    def _unique_ordered(self, array):
        '''
        returns unique elements, ordered from most common to least

        Args:
            array: np.array
                1d array like [1., 2., 3., 3., 2., ...]
        Returns:
            ordered_unique: np.array
                1d array of all unique values ordered from most common to least
            
        '''
        array = np.array(array)
    
        unique, counts = np.unique(array, return_counts=True)
    
        ordered_index = np.argsort(-counts)

        ordered_unique = unique[ordered_index]
    
        return ordered_unique

    def _resolve_labels(self, y_true, y_pred):
        '''
        resolves y_pred to choose same labeling scheme as y_true, with no match = -1


        Args:
            y_true: tensor
                true labels
            y_pred: tensor
                models predicted labels

        Returns:
            y_resolved: array
                similar to y_pred, but with labels re-named to match y_true if they match
                for example
                y_true     = [1, 2, 2, 3, 4]
                y_pred     = [2, 1, 1, 3, 3]
                y_resolved = [1, 2, 2, 3, -1]

                since this task is object encoding the label number has no intrinsic meaning
                only that 1 and 1 are the same object so label 'names' must be adjusted to 
                true labels
        '''
        y_pred     = np.array(y_pred)
        y_true     = np.array(y_true)
        y_resolved = np.full(y_pred.shape, -1)
        
        y_true_uni_ord = self._unique_ordered(y_true)
        used = set()
    
        for u in y_true_uni_ord:
            y_pred_subset = y_pred[y_true == u]
    
            y_pred_sub_uni_ord = self._unique_ordered(y_pred_subset)
    
            for unique_pred in y_pred_sub_uni_ord:
                if unique_pred not in used:
                    used.add(unique_pred)
    
                    correct = (y_true == u) & (y_pred == unique_pred)
            
                    y_resolved[correct] = u
                    break
    
        return y_resolved
    
    def _get_accuracy(self, y_true_labels, y_pred_labels):
        '''
        returns accuracy for and auto resolves labels

        Args:
            y_true_labels: tensor
                from training/valid dataset
            y_pred_labels: tensor
                output of model.label

        Returns:
            accuracy: float
                accuracy between [0, 1]
        '''
        y_pred_labels = self._resolve_labels(y_true_labels, y_pred_labels)
        
        accuracy = np.array(y_true_labels == y_pred_labels).mean()
    
        return accuracy
        
    def calc_loss(self, model_output, labels, loss_metric, accuracy_metric, decode_params):
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
        y_pred_labels = decode_to_labels(model_output, **decode_params)
               
        accuracy = self._get_accuracy(y_true_labels, y_pred_labels)        
        accuracy_metric.update_state(accuracy, sample_weight=y_pred.shape[0])
               
        return loss
        

    def calc_loss_with_margin(self, model_output, labels, loss_metric, accuracy_metric, decode_params):
        '''
        Calculates loss for siamese network

        Args:
            model_output (tensor) : shape (n_obj, encoding_depth)
            labels       (tensor) : shape (n_obj,)

        Returns:
            loss (tensor) : single scalar loss value
        
        '''
        sim_thresh = self.sim_thresh
        dif_thresh = self.dif_thresh
        
        same_obj    = self._label_mask(labels)                 # [n_labels, n_labels] bool
        cos_sim_mat = self._cosine_similiarty(model_output)    # [n_emb, n_emb] same shape as [n_labels, n_labels]

        similar   = cos_sim_mat[same_obj]
        different = cos_sim_mat[~same_obj]
        
        # sim_loss = tf.reduce_sum(sim_thresh - similar[similar < sim_thresh])
        # dif_loss = tf.reduce_sum(different[different > dif_thresh] - dif_thresh)

        # loss = sim_loss + dif_loss

        sim_loss = sim_thresh - similar[similar < sim_thresh]
        dif_loss = different[different > dif_thresh] - dif_thresh

        y_pred = tf.concat([sim_loss, dif_loss], axis=-1)
        y_pred = tf.reshape(y_pred, [-1])
        
        y_true = tf.zeros_like(y_pred)

        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        if self.parent_obj.verbose > 0:
            print(loss)

        
        if self.parent_obj.verbose > 1:
            if tf.reduce_any(tf.math.is_nan(loss)):
                print(f'''WHOOPS!
    labels
    ------
    {labels}
    
    
    similar
    -------
    {similar}
    
    different
    ---------
    {different}
    
    sim_loss
    --------
    {sim_loss}
    
    dif_loss
    --------
    {dif_loss}
    
    y_pred
    ------
    {y_pred}
    
    y_true
    ------
    {y_true}
    
    loss
    ----
    {loss}
    
    ''')

        if tf.reduce_any(tf.math.is_nan(loss)):
            loss_metric.update_state(0.0, sample_weight=y_pred.shape[0])
        else:
            loss_metric.update_state(loss, sample_weight=y_pred.shape[0])

        ##### ACCURACY #####

        y_true_labels = labels
        y_pred_labels = decode_to_labels(model_output, **decode_params)
               
        accuracy = self._get_accuracy(y_true_labels, y_pred_labels)        
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

    def _unique_ordered(self, array):
        'returns unique elements, ordered from most common to least'
        array = np.array(array)
    
        unique, counts = np.unique(array, return_counts=True)
    
        ordered_index = np.argsort(-counts)
    
        return unique[ordered_index]

    def _resolve_labels(self, y_true, y_pred):
        'resolves y_pred to choose same labeling scheme as y_true, with no match = -1'
        # print(y_pred)
        y_pred     = np.array(y_pred)
        y_true     = np.array(y_true)
        y_resolved = np.full(y_pred.shape, -1)
        
        y_true_uni_ord = self._unique_ordered(y_true)
        used = set()
    
        for u in y_true_uni_ord:
#             print(f'''
# u
# --
# {u}


# y_true_uni_ord
# --------------
# {y_true_uni_ord}

# y_pred
# ------
# {y_pred}
#             ''')
            y_pred_subset = y_pred[y_true == u]
    
            y_pred_sub_uni_ord = self._unique_ordered(y_pred_subset)
    
            for unique_pred in y_pred_sub_uni_ord:
                if unique_pred not in used:
                    used.add(unique_pred)
    
                    correct = (y_true == u) & (y_pred == unique_pred)
            
                    y_resolved[correct] = u
                    break
    
        return y_resolved
    
    def _get_accuracy(self, y_true_labels, y_pred_labels):
        y_pred_labels = self._resolve_labels(y_true_labels, y_pred_labels)
        
        accuracy = np.array(y_true_labels == y_pred_labels).mean()
    
        return accuracy
        
    def calc_loss(self, model_output, labels, loss_metric, accuracy_metric, decode_params):
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

        loss_metric.update_state(loss, sample_weight=y_pred.shape[0])

        ##### ACCURACY #####

        y_true_labels = labels
        y_pred_labels = decode_to_labels(model_output, **decode_params)
               
        accuracy = self._get_accuracy(y_true_labels, y_pred_labels)        
        accuracy_metric.update_state(accuracy, sample_weight=y_pred.shape[0])
               
        return loss

    def calc_loss_with_margin(self, model_output, labels, loss_metric, accuracy_metric, decode_params):
        '''
        Calculates loss for siamese network

        Args:
            model_output (tensor) : shape (n_obj, encoding_depth)
            labels       (tensor) : shape (n_obj,)

        Returns:
            loss (tensor) : single scalar loss value
        
        '''
        sim_thresh = .5
        dif_thresh = 2
        
        same_obj    = self._label_mask(labels)                       # [n_labels, n_labels] bool
        euc_dst_mat = self._euclidean_distance(model_output)    # [n_emb, n_emb] same shape as [n_labels, n_labels]

        similar   = euc_dst_mat[same_obj]
        different = euc_dst_mat[~same_obj]
        
        sim_loss = tf.reduce_sum(similar[similar > sim_thresh] - sim_thresh)
        dif_loss = tf.reduce_sum(dif_thresh - different[different < dif_thresh])

        loss = sim_loss + dif_loss

        loss_metric.update_state(loss, sample_weight=y_pred.shape[0])

        ##### ACCURACY #####

        y_true_labels = labels
        y_pred_labels = decode_to_labels(model_output, **decode_params)
               
        accuracy = self._get_accuracy(y_true_labels, y_pred_labels)        
        accuracy_metric.update_state(accuracy, sample_weight=y_pred.shape[0])
               
        return loss

