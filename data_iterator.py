import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def resize_image_to_multiple(image, size_mult):
    image_shape = tf.shape(image)
    h = image_shape[0]
    w = image_shape[1]
    h = (h // size_mult) * size_mult
    w = (w // size_mult) * size_mult    
    image = tf.image.resize_images(image, [h, w])
    return image

def load_input_image(path, size_mult):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = resize_image_to_multiple(image, size_mult)
    return image

def load_target_image(path, size_mult, nb_classes):
    def load_palette_image(path):
        img = Image.open(path)
        img = np.array(img, dtype=np.uint8)
        img[img==255] = 0
        return img
    image = tf.py_func(load_palette_image, [path], tf.uint8)
    image = tf.one_hot(image, nb_classes, dtype=tf.uint8)
    image.set_shape([None,None,nb_classes])
    image = resize_image_to_multiple(image, size_mult)
    return image

def get_loading_function(size_mult, nb_classes):
    def load_image_pair(input_path, target_path):
        input_image = load_input_image(input_path, size_mult)
        target_image = load_target_image(target_path, size_mult, nb_classes) 
        return input_image, target_image
    return load_image_pair

def get_pascal_dataset(image_list_file, input_dir, target_dir, size_mult, nb_classes):
    with open(image_list_file) as f:
        image_list = f.read().splitlines()
    input_files = [os.path.join(input_dir, im+'.jpg') for im in image_list]
    target_files = [os.path.join(target_dir, im+'.png') for im in image_list]
    
    nb_images = len(input_files)

    parse_function = get_loading_function(size_mult, nb_classes)

    dataset = tf.data.Dataset.from_tensor_slices((input_files, target_files))
    dataset = dataset.shuffle(nb_images)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(10)
    
    return dataset, nb_images   
