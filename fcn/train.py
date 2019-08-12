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

#pascal_dataset = data_iterator.PascalDataset(
#    'C:\\Users\\Kazim\\workspace\\segmentation\\VOC2012\\JPEGImages\\',
#    'C:\\Users\\Kazim\\workspace\\segmentation\\VOC2012\\SegmentationClass\\',
#    'C:\\Users\\Kazim\\workspace\\segmentation\\VOC2012\\ImageSets\\Segmentation\\train.txt',
#    'C:\\Users\\Kazim\\workspace\\segmentation\\VOC2012\\ImageSets\\Segmentation\\minitest.txt')

pascal_dataset = data_iterator.PascalDataset(
    '/Users/kazimpal/workspace/segmentation/VOC2012/JPEGImages',
    '/Users/kazimpal/workspace/segmentation/VOC2012/SegmentationClass',
    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/train.txt',
    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/minitest.txt')

for e in range(5):
    for k in range(1, pascal_dataset.nb_images_val + 1):
        X, y = next(pascal_dataset.val_gen)
        print(X.shape, y.shape)
        plt.imshow(X[0,:,:,:].astype(np.uint8))
        plt.show()

X = tf.placeholder(tf.float32, [None, None, None, 3])
y = tf.placeholder(tf.float32, [None, None, None, nb_classes])
y_pred, vgg_out, conv7_score = models.fcn_32(X, nb_classes)

y_flat = tf.keras.layers.Reshape((-1, nb_classes))(y)
y_pred_flat = tf.keras.layers.Reshape((-1, nb_classes))(y_pred)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y_flat, logits=y_pred_flat)

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "seg")
print(train_vars)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss, var_list=train_vars)

with K.get_session() as sess:
    print('running global_variables_initializer')
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.variables_initializer (train_vars))

    print('starting training ... ')
    for epoch in range(1, epochs + 1):
        
        for i in range(1, pascal_dataset.nb_images_train + 1):
            Xi, yi = next(pascal_dataset.train_gen)
            _, l, yp, yt, yf = sess.run([optimizer, loss, y_pred, y, y_flat], feed_dict={X:Xi, y:yi})

            foo = np.argmax(yp, axis=-1)
            foo = np.squeeze(foo)

            bar = np.argmax(yt, axis=-1)
            bar = np.squeeze(bar)

            mun = np.argmax(yf, axis=-1)
            print(np.amax(mun))

            plt.subplot(121)
            plt.imshow(foo)
            plt.subplot(122)
            plt.imshow(bar)
            plt.show()
            #for c in range(21):
            #    foo = yp[0,:,:,c]
            #    plt.imshow(foo)
            #    plt.show()
            
            #l2 = np.squeeze(l)
            #plt.imshow(l2)
            #plt.show()
           
            yp = np.argmax(yp, axis=-1)
            yp_min = np.amin(yp)
            yp_max = np.amax(yp)

            '''print(yf.shape, ypf.shape)
            print(np.amin(yf), np.amax(yf))
            print(np.amin(ypf), np.amax(ypf))
            print(l.shape, np.amin(l), np.amax(l))
            input()
            count = 0
            for k in range(ypf.shape[1]):
                foo = yf[0,k,:]
                bar = ypf[0,k,:]
                if foo[0] == 1:
                    continue
                print(foo)
                print(bar)
                input()
                count += 1
                if count > 3:
                    break'''
            
            l = np.mean(l)
            print('Epoch {}, Batch {} of {}, Loss {:.3f}, yp_range {} {}'.format(epoch, i, pascal_dataset.nb_images_train, l, yp_min, yp_max))
        
        val_loss = 0
        for i in range(1, pascal_dataset.nb_images_val + 1):
            Xi, yi = next(pascal_dataset.val_gen)
            l, pi, yi = sess.run([loss, y_pred, y], feed_dict={X:Xi, y:yi})
            l = np.mean(l)
            val_loss += l * 1.0/pascal_dataset.nb_images_val
            
            pred_path = "test_{}_epoch_{}.png".format(i, epoch)
            save_y_pred(pi, pred_path)

            truth_path = "truth_{}.png".format(i)
            save_y_pred(yi, truth_path)

        print('Epoch {}, Val Loss {:.3f}'.format(epoch, val_loss))
                    
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
