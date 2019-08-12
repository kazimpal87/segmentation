import tensorflow as tf
import tensornets as nets
import numpy as np

def get_vgg16_pooling_layer(layer_num):
    name = "vgg16/conv{}/pool/MaxPool:0".format(layer_num)
    output = tf.get_default_graph().get_tensor_by_name(name)
    return output

def conv2d_helper(inputs, filters, kernel_size, name, do_dropout=True, activation='relu'):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, name='seg/'+name)(inputs)
    if do_dropout:
        x = tf.layers.dropout(x, name='seg/'+name+'_dropout')
    return x

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def fcn_32(inputs, nb_classes):
    vgg16_pretrained = tf.keras.applications.VGG16(include_top=False, input_tensor=inputs, weights='imagenet')
    pool5 = vgg16_pretrained.output

    # Replace fc layers with conv layers
    conv6 = tf.keras.layers.Conv2D(4096, (7, 7), padding='same', activation='relu', name='seg/conv6')(pool5)
    drop6 = tf.keras.layers.Dropout(rate=0.5)(conv6)
    
    conv7 = tf.keras.layers.Conv2D(4096, (1, 1), padding='same', activation='relu', name='seg/conv7')(drop6)
    drop7 = tf.keras.layers.Dropout(rate=0.5)(conv7)
    
    conv7_score = tf.keras.layers.Conv2D(nb_classes, (1, 1), padding='same', activation='linear', name='seg/conv7_score')(drop7)
    
    final_score = tf.keras.layers.Conv2DTranspose(
        filters=nb_classes, 
        kernel_size=(64, 64),
        strides=(32, 32),
        padding='same',
        activation=None,
        name='seg/final_score',
        kernel_initializer=tf.keras.initializers.Constant(bilinear_upsample_weights(32, nb_classes)))(conv7_score)
    
    return final_score, pool5, conv7_score

def fcn_16(sess, inputs, nb_classes):    
    vgg16_pretrained = nets.VGG16(inputs, is_training=True, stem=True)
    sess.run(vgg16_pretrained.pretrained())
    
    # Replace fc layers with conv layers
    conv6 = conv2d_helper(vgg16_pretrained, 4096, (7, 7), name='conv6')
    conv7 = conv2d_helper(conv6, 4096, (1, 1), name='conv7')
    conv7_score = conv2d_helper(conv7, nb_classes, (1, 1), name='conv7_score', do_dropout=False)

    # Upsample conv7    
    conv7_score_x2 = upsample_helper(conv7_score, nb_classes, 2, name='conv7_score_x2')

    # Get pool4 and class scores
    pool4 = get_vgg16_pooling_layer(4)
    pool4_score = conv2d_helper(pool4, nb_classes, (1, 1), name='pool4_score')

    # Compute sum
    pool4_conv7_score = tf.math.add(pool4_score, conv7_score_x2)

    # Upsample sum
    final_score = upsample_helper(pool4_conv7_score, nb_classes, 16, name='final_score')
    return final_score

def fcn_8(sess, inputs, nb_classes):
    vgg16_pretrained = nets.VGG16(inputs, is_training=True, stem=True)
    sess.run(vgg16_pretrained.pretrained())
    
    # Replace fc layers with conv layers
    conv6 = conv2d_helper(vgg16_pretrained, 4096, (7,7), name='conv6')
    conv7 = conv2d_helper(conv6, 4096, (1,1), name='conv7')
    conv7_score = conv2d_helper(conv7, nb_classes, (1,1), name='conv7_score', do_dropout=False)

    # Upsample conv7    
    conv7_score_x4 = upsample_helper(conv7_score, nb_classes, 4, name='conv7_score_x4')

    # Get pool4 and class scores
    pool4 = get_vgg16_pooling_layer(4)
    pool4_score = conv2d_helper(pool4, nb_classes, (1,1), name='pool4_score')
    pool4_score_x2 = upsample_helper(pool4_score, nb_classes, 2, name='pool4_score_x2')

    # Get pool3 and class scores
    pool3 = get_vgg16_pooling_layer(3)
    pool3_score = conv2d_helper(pool3, nb_classes, (1,1), name='pool3_score')

    # Compute sum
    sum_score = tf.math.add_n([conv7_score_x4, pool4_score_x2, pool3_score])

    # Upsample sum
    final_score = upsample_helper(sum_score, nb_classes, 8, name='final_score')
    return final_score
