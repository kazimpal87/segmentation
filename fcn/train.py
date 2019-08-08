import tensorflow as tf
from segmentation import data_iterator
from segmentation.fcn import models

sess = tf.Session()
nb_classes = 21
epochs = 100
save_model_path = 'test'

dataset, nb_images = data_iterator.get_pascal_dataset(
    '/Users/kazimpal/workspace/segmentation/VOC2012/ImageSets/Segmentation/train.txt', 
    '/Users/kazimpal/workspace/segmentation/VOC2012/JPEGImages/', 
    '/Users/kazimpal/workspace/segmentation/VOC2012/SegmentationClass/', 
    32, 21)
iter = dataset.make_one_shot_iterator()
X, labels = iter.get_next()

model = models.fcn_32(sess, X, 21)
print(model)
input()

loss = tf.losses.softmax_cross_entropy(labels, model)

optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

with sess.as_default():
    print('running global_variables_initializer')
    sess.run(tf.global_variables_initializer())

    print('starting training ... ')
    for epoch in range(1, epochs + 1):
        for i in range(1, nb_images + 1):
            l, _ = sess.run([loss, optimizer])
            if i % 1 == 0:
                print('Epoch {}, Batch {}, Loss {:.3f}'.format(epoch, i, l), end='')
                    
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)