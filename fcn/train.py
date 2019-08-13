import numpy as np
import tensorflow as tf
from segmentation import data_iterator
from segmentation.fcn import models
import imageio
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

nb_classes = 21
epochs = 100
save_model_path = 'test'

def color_map(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype='uint8')
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap

cmap = color_map(21)

def save_y_pred(pred, path):
    print(pred.shape)
    print(pred.dtype, np.amin(pred), np.amax(pred))
    
    pred = np.argmax(pred, axis=-1)
    print(pred.shape)
    print(pred.dtype, np.amin(pred), np.amax(pred))
    
    pred = np.squeeze(cmap[pred])
    print(pred.shape)
    print(pred.dtype, np.amin(pred), np.amax(pred))
    
    imageio.imsave(path, pred)
    print(path)

pascal_dataset = data_iterator.PascalDataset(
    'C:\\Users\\Kazim\\workspace\\segmentation\\VOC2012\\JPEGImages\\',
    'C:\\Users\\Kazim\\workspace\\segmentation\\VOC2012\\SegmentationClass\\',
    'C:\\Users\\Kazim\\workspace\\segmentation\\VOC2012\\ImageSets\\Segmentation\\train.txt',
    'C:\\Users\\Kazim\\workspace\\segmentation\\VOC2012\\ImageSets\\Segmentation\\minitest.txt')

#pascal_dataset = data_iterator.PascalDataset(
#    '/Users/kazimpal/workspace/segmentation/VOC2012/JPEGImages',
#    '/Users/kazimpal/workspace/segmentation/VOC2012/SegmentationClass',
#    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/train.txt',
#    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/minitest.txt')

X = tf.placeholder(tf.float32, [None, None, None, 3])
y = tf.placeholder(tf.float32, [None, None, None, nb_classes])
y_pred = models.fcn_8(X, nb_classes)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "seg")
train_op = optimizer.minimize(loss, var_list=train_vars)

with K.get_session() as sess:
    print('running global_variables_initializer')
    sess.run(tf.variables_initializer (train_vars))
    sess.run(tf.variables_initializer (optimizer.variables()))

    print('starting training ... ')
    for epoch in range(1, epochs + 1):
        
        for i in range(1, pascal_dataset.nb_images_train + 1):
            Xi, yi = next(pascal_dataset.train_gen)
            _, l, yp = sess.run([train_op, loss, y_pred], feed_dict={X:Xi, y:yi})
            yp = np.argmax(yp, axis=-1)
            yt = np.argmax(yi, axis=-1)
            yt_max = np.amax(yt)
            yp_max = np.amax(yp)
            print('Epoch {}, Batch {} of {}, Loss {:.3f}, yp_range {} {}'.format(epoch, i, pascal_dataset.nb_images_train, l, yt_max, yp_max))
        
        val_loss = 0
        for i in range(1, pascal_dataset.nb_images_val + 1):
            Xi, yi = next(pascal_dataset.val_gen)
            l, pi, yi = sess.run([loss, y_pred, y], feed_dict={X:Xi, y:yi})
            
            val_loss += l * 1.0/pascal_dataset.nb_images_val
            
            pred_path = "test_{}_epoch_{}.png".format(i, epoch)
            save_y_pred(pi, pred_path)

            truth_path = "truth_{}.png".format(i)
            save_y_pred(yi, truth_path)

        print('Epoch {}, Val Loss {:.3f}'.format(epoch, val_loss))
                    
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
