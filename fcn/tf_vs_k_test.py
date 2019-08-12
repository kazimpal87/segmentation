import numpy as np
import tensorflow as tf
from segmentation import data_iterator
from segmentation.fcn import models
import matplotlib.pyplot as plt
import imageio
from tensorflow.keras import backend as K

nb_classes = 21
epochs = 100
save_model_path = 'test'

pascal_dataset = data_iterator.PascalDataset(
    '/Users/kazimpal/workspace/segmentation/VOC2012/JPEGImages',
    '/Users/kazimpal/workspace/segmentation/VOC2012/SegmentationClass',
    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/train.txt',
    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/minitest.txt')

def pred_keras(img):
    vgg16_pretrained = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    result = vgg16_pretrained.predict(img)
    return result

def pred_tf(img):
    X = tf.placeholder(tf.float32, [None, None, None, 3])
    vgg16_pretrained = tf.keras.applications.VGG16(include_top=False, input_tensor=X, weights='imagenet')
    pool5 = vgg16_pretrained.output

    with K.get_session() as sess:
        result = sess.run(pool5, feed_dict={X:img})
    return result

for k in range(5):
    X, y = next(pascal_dataset.train_gen)
    #plt.imshow(X[0,:,:,:])
    #plt.show()
    result_k = pred_keras(X)
    result_tf = pred_tf(X)

    for c in range(20):
        print(result_k[0,:,:,c])
        print(result_tf[0,:,:,c])
        input()
