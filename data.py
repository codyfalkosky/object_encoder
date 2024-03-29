import tensorflow as tf


class Data:
    '''
    Stores and pre-processes a dataset for training object encoding

    Example:
        >>> dataset = Data([path/to/record_1.tfr, path/to.record_2.tfr]).dataset
    
    '''
    
    def __init__(self, records_path_list):
        self.records_path_list = records_path_list
        self._build_dataset()

    @staticmethod
    def parse_tfrecord_fn(serialized_example):
        '''
        takes a seralized tfr and parses to objects, labels, coords dictionary

        Args:
            serialized_example (byte stream?) : a single entry in a TFRecord file

        Returns:
            example (dictionary) : like {'objects':objects, 'coords':coords, 'labels':labels}
        '''
        feature_description = {
            'objects' : tf.io.FixedLenFeature([], tf.string),
            'coords'  : tf.io.FixedLenFeature([], tf.string),
            'labels'  : tf.io.FixedLenFeature([], tf.string),
        }
        
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # parse all serialized tensors
        objects = tf.io.parse_tensor(example['objects'], tf.uint8)
        coords  = tf.io.parse_tensor(example['coords'],  tf.float32)
        labels  = tf.io.parse_tensor(example['labels'],  tf.uint8)

        # clips to float and normalize
        objects  = tf.cast(objects, tf.float32)
        objects /= 255.

        # labels to float
        labels = tf.cast(labels, tf.float32)       

        return {'objects':objects, 'coords':coords, 'labels':labels}

    def _build_dataset(self):
        self.dataset = tf.data.TFRecordDataset(self.records_path_list)
        self.dataset = self.dataset.map(self.parse_tfrecord_fn)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.shuffle(buffer_size=1024)
        


if __name__ == '__main__':
    dataset = Data(['/Users/codyfalkosky/Desktop/Siamese_Data_TFRecord/examples_3240.tfr'])
    
    for batch in dataset.dataset:
        break
