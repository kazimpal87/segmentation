import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import imageio
from random import shuffle

class PascalDataset(object):

    def __init__(self, image_dir, seg_dir, file_list_train, file_list_val, size_mult=32, nb_classes=21):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.file_list_train = file_list_train
        self.file_list_val = file_list_val
        self.size_mult = size_mult
        self.nb_classes = nb_classes

        self.make_dataset_and_iterator()

    def resize_image_to_multiple(self, image):
        image_shape = tf.shape(image)
        h = image_shape[0]
        w = image_shape[1]
        h = (h // self.size_mult) * self.size_mult
        w = (w // self.size_mult) * self.size_mult
        image = tf.image.resize_images(image, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image

    def load_input_image(self, path):
        img = Image.open(path)
        w, h = img.size
        img = img.resize(((w//32)*32, (h//32)*32))
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def load_label_image(self, path):
        img = Image.open(path)
        w, h = img.size
        img = img.resize(((w//32)*32, (h//32)*32))
        img = np.array(img, dtype=np.uint8)
        img[img==255] = 0
        y = np.zeros((1, img.shape[0], img.shape[1], self.nb_classes), dtype=np.float32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                y[0, i, j, img[i][j]] = 1
        return y

    def make_dataset_and_iterator(self):
        with open(self.file_list_train) as f:
            self.train_files = f.read().splitlines()
        with open(self.file_list_val) as f:
            self.val_files = f.read().splitlines()

        self.train_gen = self.batch_generator(self.train_files)
        self.val_gen = self.batch_generator(self.val_files)

        self.nb_images_train = len(self.train_files)
        self.nb_images_val = len(self.val_files)
    
    def batch_generator(self, data, do_shuffle=False):
        if do_shuffle: shuffle(data)

        count = 0
        while True:
            if count >= len(data):
                count = 0
                if do_shuffle: shuffle(data)

            f_i = data[count]
            src_img = self.load_input_image(os.path.join(self.image_dir, f_i+'.jpg'))
            lab_img = self.load_label_image(os.path.join(self.seg_dir, f_i+'.png'))
            count += 1
            yield src_img, lab_img
