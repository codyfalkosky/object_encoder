import tensorflow as tf
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from boxes import Boxes
from IPython.display import clear_output

parent_glob = '/Users/codyfalkosky/Desktop/Siamese_Data/*'
seq_folders = glob.glob(parent_glob)


# +
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()  # get its value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(clips, labels, coords):
    """Reads a single (image, label) example and serializes for storage as TFRecord"""

    feature = {
        'image' : _bytes_feature(tf.io.serialize_tensor(clips)),
        'label' : _bytes_feature(tf.io.serialize_tensor(labels)),
        'label' : _bytes_feature(tf.io.serialize_tensor(coords))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# -

class DataBuilder:

    'for converting raw img, obj_labels to clips, labels, coords'

    def __init__(self, path):
        self.path = path

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

        return labels, coords

    def _coords_to_clip(self, coords):
        'converts relative float coords to absolute int coords'
        coords = Boxes.scale(coords, 416, 416)
        coords = Boxes.convert(coords, 'cxcywh_xyxy')
        coords = tf.round(coords)
        coords = tf.cast(coords, tf.int32)
        return coords

    def _clip_and_resize(self, img, coords):
        'clips coords from img and resizes'
        x1 = coords[0]
        y1 = coords[1]
        x2 = coords[2]
        y2 = coords[3]
        
        clip = img[y1:y2, x1:x2, :]
        clip = tf.image.resize(clip, [34, 42])
        return clip

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
        clip_coords    = self._coords_to_clip(coords)
        img            = self._image_to_tensor(img_path)

        clips  = []

        for coord in clip_coords:
            clip = self._clip_and_resize(img, coord)
            clips.append(clip)

        clips = tf.stack(clips, axis=0)
        return clips, labels, coords

    def _build(self):
        clips_list  = []
        labels_list = []
        coords_list = []

        for img_path, ann_path in zip(self.imgs, self.anot):
            
            clips, labels, coords = self._compile_single_annotation(img_path, ann_path)
            
            clips_list.append(clips)
            labels_list.append(labels)
            coords_list.append(coords)

        clips  = tf.concat(clips_list,  axis=0)
        labels = tf.concat(labels_list, axis=0)
        coords = tf.concat(coords_list, axis=0)

        self.clips  = clips
        self.labels = labels
        self.coords = coords

    def serialize(self):
        'export example as TFRecord'
        example = serialize_example(self.clips, self.labels, self.coords)

        return example


save_file = '/Users/codyfalkosky/Desktop/Siamese_Data_TFRecord/examples.tfr'

# +
writer = tf.io.TFRecordWriter(shard_filename)

for folder in seq_folders:
    data = Data(folder)

    example = 


# +
# serialize_example(data.clips, data.labels, data.coords)

# +
# root = '/Users/codyfalkosky/Desktop/Siamese_Data_TFRecord'

for i, folder in enumerate(seq_folders):
    parent_folder = root + f'/example_{i}'
    if os.path.exists(parent_folder):
        continue
    
    print(f'Building Examples at {folder}')
    data = DataBuilder(folder)
    
    os.mkdir(parent_folder)
    print(f'\n    saving at {parent_folder}\n\n')

    np.save(parent_folder + '/clips.npy', data.clips)
    np.save(parent_folder + '/labels.npy', data.labels)
    np.save(parent_folder + '/coords.npy', data.coords)
# -

root = '/Users/codyfalkosky/Desktop/Siamese_Data_TFRecord'
