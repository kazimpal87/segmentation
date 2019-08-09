import numpy as np
import tensorflow as tf
from segmentation import data_iterator
from segmentation.fcn import models
import imageio

sess = tf.Session()
nb_classes = 21
epochs = 100
save_model_path = 'test'
fcn_version = 32

version_to_model = {
    8: models.fcn_8,
    16: models.fcn_16,
    32: models.fcn_32
}

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

def save_prediction(pred, path):
    print(pred.dtype, np.amin(pred), np.amax(pred))
    pred = np.argmax(pred, axis=-1)
    print(pred.dtype, np.amin(pred), np.amax(pred))
    pred = np.squeeze(cmap[pred])
    print(pred.dtype, np.amin(pred), np.amax(pred))
    imageio.imsave(path, pred)
    print(path)

pascal_dataset = data_iterator.PascalDataset(
    '/Users/kazimpal/workspace/segmentation/VOC2012/JPEGImages/',
    '/Users/kazimpal/workspace/segmentation/VOC2012/SegmentationClass/', 
    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/train.txt',
    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/minitest.txt')
X, labels = pascal_dataset.iterator.get_next()

prediction = version_to_model[fcn_version](sess, X, 21)

loss = tf.losses.softmax_cross_entropy(labels, prediction)

optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

with sess.as_default():
    print('running global_variables_initializer')
    sess.run(tf.global_variables_initializer())

    print('starting training ... ')
    for epoch in range(1, epochs + 1):
        
        print('initializing iterator to training ... ')
        sess.run(pascal_dataset.iterator.initializer, feed_dict= {pascal_dataset.img_ph: pascal_dataset.img_files_train, pascal_dataset.seg_ph:pascal_dataset.seg_files_train})
        for i in range(1, pascal_dataset.nb_images_train + 1):
            l, _ = sess.run([loss, optimizer])
            if i % 10 == 0:
                print('Epoch {}, Batch {} of {}, Loss {:.3f}'.format(epoch, i, pascal_dataset.nb_images_train, l))
        
        print('initializing iterator to validation ... ')
        sess.run(pascal_dataset.iterator.initializer, feed_dict= {pascal_dataset.img_ph: pascal_dataset.img_files_val, pascal_dataset.seg_ph:pascal_dataset.seg_files_val})
        val_loss = 0
        for i in range(1, pascal_dataset.nb_images_val + 1):
            l, pred = sess.run([loss, prediction])
            val_loss += l * 1.0/pascal_dataset.nb_images_val
            pred_path = "test_{}_epoch_{}.png".format(i, epoch)
            save_prediction(pred, pred_path)
        print('Epoch {}, Val Loss {:.3f}'.format(epoch, val_loss))
        
                    
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
