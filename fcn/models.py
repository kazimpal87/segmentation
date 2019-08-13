import tensorflow as tf
import numpy as np

def conv2d_helper(inputs, filters, kernel_size, name, do_dropout=True, activation='relu'):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation, name='seg/'+name)(inputs)
    if do_dropout:
        x = tf.keras.layers.Dropout(rate=0.5, name='seg/'+name+'_dropout')(x)
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

def upsample_helper(inputs, filters, scale, name):
    x = tf.keras.layers.Conv2DTranspose(
        filters,
        (scale*2, scale*2),
        strides=(scale, scale),
        padding='same',
        activation='linear',
        name='seg/'+name,
        kernel_initializer=tf.keras.initializers.Constant(bilinear_upsample_weights(scale, filters)))(inputs)
    return x

def fcn_32(inputs, nb_classes):
    vgg16_pretrained = tf.keras.applications.VGG16(include_top=False, input_tensor=inputs, weights='imagenet')
    pool5 = vgg16_pretrained.get_layer('block5_pool').output
    conv6 = conv2d_helper(pool5, 4096, (7, 7), name='conv6')
    conv7 = conv2d_helper(conv6, 4096, (1, 1), name='conv7')
    conv8 = conv2d_helper(conv7, nb_classes, (1, 1), name='conv8', activation='linear', do_dropout=False)
    logits = upsample_helper(conv8, nb_classes, 32, name='logits')
    return logits

def fcn_16(inputs, nb_classes):
    vgg16_pretrained = tf.keras.applications.VGG16(include_top=False, input_tensor=inputs, weights='imagenet')
    pool4 = vgg16_pretrained.get_layer('block4_pool').output
    pool5 = vgg16_pretrained.get_layer('block5_pool').output
    print(pool4)
    print(pool5)
    
    conv6 = conv2d_helper(pool5, 4096, (7, 7), name='conv6')
    conv7 = conv2d_helper(conv6, 4096, (1, 1), name='conv7')
    conv8 = conv2d_helper(conv7, nb_classes, (1, 1), name='conv8', activation='linear', do_dropout=False)
    print(conv6)
    print(conv7)
    print(conv8)

    # Upsample conv8
    conv8_up = upsample_helper(conv8, nb_classes, 2, name='conv8_up')
    print(conv8_up)

    # Get pool4 and class scores
    pool4_scores = conv2d_helper(pool4, nb_classes, (1, 1), name='pool4_scores', activation='linear', do_dropout=False)
    print(pool4_scores)

    # Compute sum
    sum1 = tf.math.add(pool4_scores, conv8_up, name='sum1')
    print(sum1)

    # Upsample sum
    logits = upsample_helper(sum1, nb_classes, 16, name='logits')
    print(logits)
    input()

    return logits

def fcn_8(inputs, nb_classes):
    vgg16_pretrained = tf.keras.applications.VGG16(include_top=False, input_tensor=inputs, weights='imagenet')
    pool3 = vgg16_pretrained.get_layer('block3_pool').output
    pool4 = vgg16_pretrained.get_layer('block4_pool').output
    pool5 = vgg16_pretrained.get_layer('block5_pool').output
    print(pool3)
    print(pool4)
    print(pool5)
    
    # Replace fc layers with conv layers
    conv6 = conv2d_helper(pool5, 4096, (7, 7), name='conv6')
    conv7 = conv2d_helper(conv6, 4096, (1, 1), name='conv7')
    conv7_score = conv2d_helper(conv7, nb_classes, (1, 1), name='conv7_score', activation='linear', do_dropout=False)
    print(conv6)
    print(conv7)
    print(conv7_score)

    # Upsample conv7    
    conv7_score_x4 = upsample_helper(conv7_score, nb_classes, 4, name='conv7_score_x4')
    print(conv7_score_x4)

    # Get pool4 and class scores
    pool4_score = conv2d_helper(pool4, nb_classes, (1,1), name='pool4_score', activation='linear', do_dropout=False)
    pool4_score_x2 = upsample_helper(pool4_score, nb_classes, 2, name='pool4_score_x2')
    print(pool4_score)
    print(pool4_score_x2)

    # Get pool3 and class scores
    pool3_score = conv2d_helper(pool3, nb_classes, (1,1), name='pool3_score', activation='linear', do_dropout=False)
    print(pool3_score)

    # Compute sum
    sum_score = tf.math.add_n([conv7_score_x4, pool4_score_x2, pool3_score])
    print(sum_score)

    # Upsample sum
    logits = upsample_helper(sum_score, nb_classes, 8, name='logits')
    print(logits)

    return logits

