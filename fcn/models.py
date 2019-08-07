import tensorflow as tf
import tensornets as nets

def fcn_32(sess, inputs, nb_classes):    
    vgg16_pretrained = nets.VGG16(inputs, is_training=True, stem=True)
    sess.run(vgg16_pretrained.pretrained())
    print(vgg16_pretrained)
    x = tf.layers.conv2d(vgg16_pretrained, filters=4096, kernel_size=(7,7), padding='same', activation='relu', name='fc6')
    print(x)
    x = tf.layers.dropout(x)
    print(x)
    x = tf.layers.conv2d(x, filters=4096, kernel_size=(1,1), padding='same', activation='relu', name='fc7')
    print(x)
    x = tf.layers.dropout(x)
    print(x)
    x = tf.layers.conv2d(x, filters=nb_classes, kernel_size=(1,1), padding='same', activation='relu', name='score1')
    print(x)
    x = tf.layers.conv2d_transpose(x, filters=nb_classes, kernel_size=(32,32), strides=(32,32), padding='valid', activation=None, name='score2')
    print(x)
    return x, vgg16_pretrained
