import tensorflow as tf
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from boxes import Boxes
from IPython.display import clear_output
from tqdm.notebook import tqdm
tf.keras.utils.set_random_seed(2)
tf.config.experimental.enable_op_determinism()

parent_glob = '/Users/codyfalkosky/Desktop/Obj_Encoder_Data/Obj_Encoder_Data_New/*'
seq_folders = glob.glob(parent_glob)


# +
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()  # get its value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(objects, coords, labels):
    """Reads a single (image, label) example and serializes for storage as TFRecord"""

    objects  = objects * 255.
    objects  = tf.cast(objects, tf.uint8)

    labels   = tf.cast(labels, tf.uint8)

    feature = {
        'objects' : _bytes_feature(tf.io.serialize_tensor(objects)),
        'coords'  : _bytes_feature(tf.io.serialize_tensor(coords)),
        'labels'  : _bytes_feature(tf.io.serialize_tensor(labels))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# -

class DataBuilder:

    'for converting raw img, obj_labels to objs, labels, coords'

    def __init__(self, path, distortion=None):
        self.path = path
        self.distortion = distortion

        # annotation paths
        self.anot = self._get_all_anot_paths(path)
        self.anot.sort()

        # image paths
        self.imgs = self._get_all_img_paths(self.anot)

        # build
        self._build()

    def _image_to_tensor(self, image_path):
        'reads in an image from path'
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img)
        img = tf.image.resize(img, [416, 416])
        img /= 255.
        
        return img

    def _annotation_to_tensors(self, annotation_path):
        'reads in an annotation from path'
        file = tf.io.read_file(annotation_path)
        file = tf.strings.split(file)
        file = tf.strings.to_number(file)
        file = tf.reshape(file, (-1, 5))
        
        labels = file[:, 0]
        coords = file[:, 1:5]

        if self.distortion:
            d0  = coords.shape[0]
            x_d = tf.random.uniform([d0, 1], self.distortion['x_min'], self.distortion['x_max'])
            y_d = tf.random.uniform([d0, 1], self.distortion['y_min'], self.distortion['y_max'])
            h_d = tf.random.uniform([d0, 1], self.distortion['h_min'], self.distortion['h_max'])
            w_d = tf.random.uniform([d0, 1], self.distortion['w_min'], self.distortion['w_max'])
            
            dist_matrix = tf.concat([x_d, y_d, h_d, w_d], axis=1)
            coords *= dist_matrix

        return labels, coords

    def _coords_to_extract(self, coords):
        'converts relative float coords to absolute int coords'
        coords = Boxes.scale(coords, 416, 416)
        coords = Boxes.convert(coords, 'cxcywh_xyxy')
        coords = tf.clip_by_value(coords, 0, 416)
        coords = tf.round(coords)
        coords = tf.cast(coords, tf.int32)
        return coords

    def _extract_and_resize(self, img, coords):
        'extracts obj from img and resizes'
        x1 = coords[0]
        y1 = coords[1]
        x2 = coords[2]
        y2 = coords[3]
        
        obj = img[y1:y2, x1:x2, :]
        obj = tf.image.resize(obj, [32, 40])
        return obj

    def _annot_path_to_img_path(self, ann_path):
        'converts annotation path to relative image path'
        img_path = ann_path.replace('annotations', 'images').replace('txt', 'JPEG')
        return img_path

    def _get_all_anot_paths(self, path):
        'returns list of all annotation paths from a parent folder path'
        all_anot_paths = glob.glob(path + '/annotations/*')
        return all_anot_paths

    def _get_all_img_paths(self, anot_paths):
        'returns list of all image paths from a parend folder path'
        img_paths = [self._annot_path_to_img_path(anot_path) for anot_path in anot_paths]
        return img_paths

    def _compile_single_annotation(self, img_path, ann_path):
        labels, coords = self._annotation_to_tensors(ann_path)
        extract_coords = self._coords_to_extract(coords)
        img            = self._image_to_tensor(img_path)

        objs  = []

        for coord in extract_coords:
            obj = self._extract_and_resize(img, coord)
            objs.append(obj)

        objs = tf.stack(objs, axis=0)
        return objs, labels, coords

    def _build(self):
        objs_list  = []
        labels_list = []
        coords_list = []

        for img_path, ann_path in zip(self.imgs, self.anot):
            
            objs, labels, coords = self._compile_single_annotation(img_path, ann_path)
            
            objs_list.append(objs)
            labels_list.append(labels)
            coords_list.append(coords)

        objs   = tf.concat(objs_list,   axis=0)
        labels = tf.concat(labels_list, axis=0)
        coords = tf.concat(coords_list, axis=0)

        self.objs   = objs
        self.labels = labels
        self.coords = coords

    def serialize(self):
        'export example as TFRecord'
        example = serialize_example(self.objs, self.coords, self.labels)
        
        return example


if __name__ == '__main__':
    save_file = '/Users/codyfalkosky/Desktop/Obj_Encoder_Data/TFRecords_Valid/obj_encoder_valid_new.tfr'
    writer = tf.io.TFRecordWriter(save_file)
    
    print('Serializing Data')
    
    # distortion = {
    #     'x_min': .96, 'x_max': 1.04,  'y_min':.96, 'y_max':1.04,
    #     'h_min':  1,  'h_max': 1.4,   'w_min': 1,  'w_max':1.4
    # }
    
    for folder in tqdm(seq_folders):
    
        data    = DataBuilder(folder)
        example = data.serialize()
    
        writer.write(example)
    
    writer.close()
# ## SHOW EXTRACTIONS

# +
# distortion = {
#     'x_min': .96, 'x_max': 1.04,  'y_min':.96, 'y_max':1.04,
#     'h_min':  1,  'h_max': 1.4,   'w_min': 1,  'w_max':1.4
# }

# data = DataBuilder(seq_folders[1], distortion=distortion)

# +
# idxs = np.random.randint(0, data.objs.shape[0], 24)

# plt.figure(figsize=(10, 5))

# for i in range(24):
#     plt.subplot(4, 6, i+1)
#     plt.imshow(data.objs[idxs[i]])
#     plt.axis('off')

# plt.show()
# -


