import os
import numpy as np
from PIL import Image
import tensorflow as tf

class PascalDataset(object):

    def __init__(self, image_dir, seg_dir, file_list_train, file_list_val, size_mult=32, nb_classes=21):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.file_list_train = file_list_train
        self.file_list_val = file_list_val
        self.size_mult = size_mult
        self.nb_classes = nb_classes

        self.make_dataset_and_iterator()
        self.make_file_lists()

    def resize_image_to_multiple(self, image):
        image_shape = tf.shape(image)
        h = image_shape[0]
        w = image_shape[1]
        h = (h // self.size_mult) * self.size_mult
        w = (w // self.size_mult) * self.size_mult    
        image = tf.image.resize_images(image, [h, w])
        return image

    def load_input_image(self, path):
        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = self.resize_image_to_multiple(image)
        return image

    def load_target_image(self, path):

        def load_palette_image(path):
            img = Image.open(path)
            img = np.array(img, dtype=np.uint8)
            img[img==255] = 0
            return img

        image = tf.py_func(load_palette_image, [path], tf.uint8)
        image = tf.one_hot(image, self.nb_classes, dtype=tf.uint8)
        image.set_shape([None, None, self.nb_classes])
        image = self.resize_image_to_multiple(image)
        return image

    def get_loading_function(self):
        
        def load_image_pair(input_path, target_path):
            input_image = self.load_input_image(input_path)
            target_image = self.load_target_image(target_path) 
            return input_image, target_image
        return load_image_pair

    def make_dataset_and_iterator(self):
        parse_function = self.get_loading_function()
        self.img_ph = tf.placeholder(tf.string, [None,])
        self.seg_ph = tf.placeholder(tf.string, [None,])
        
        self.dataset = tf.data.Dataset.from_tensor_slices((self.img_ph, self.seg_ph))
        self.dataset = self.dataset.map(parse_function)
        self.dataset = self.dataset.batch(1)
        self.dataset = self.dataset.prefetch(10)

        self.iterator = self.dataset.make_initializable_iterator()

    def make_file_lists(self):

        def make_one_list(file_list):
            with open(file_list) as f:
                files = f.read().splitlines()
            img_files = [os.path.join(self.image_dir, im+'.jpg') for im in files]
            seg_files = [os.path.join(self.seg_dir, im+'.png') for im in files]
            return img_files, seg_files

        self.img_files_train, self.seg_files_train = make_one_list(self.file_list_train)
        self.img_files_val,  self.seg_files_val = make_one_list(self.file_list_val)
        
        self.nb_images_train = len(self.img_files_train)
        self.nb_images_val = len(self.img_files_val)

    
