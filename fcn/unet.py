import tensorflow as tf
import numpy as np

def conv_3x3_relu(inputs, filters, name):
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu', name=name)(inputs)
    return x

def conv_block(inputs, filters_list, name):
    x = inputs
    for k, filters in enumerate(filters_list):
        x = conv_3x3_relu(x, filters, '{}/conv{}'.format(name, k))
    return x

def downsample(inputs, name):
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name=name)(inputs)
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

def upsample(inputs, filters, name):
    x = tf.keras.layers.Conv2DTranspose(
        filters,
        (2, 2),
        strides=(2, 2),
        padding='valid',
        activation='relu',
        name=name)(inputs)
        #kernel_initializer=tf.keras.initializers.Constant(bilinear_upsample_weights(2, filters)))(inputs)
    return x

def copy_and_crop(left, right):
    left_dims = left.shape.as_list()[1:3]
    right_dims = right.shape.as_list()[1:3]
    cropping = ( (left_dims[0] - right_dims[0]) // 2,  (left_dims[1] - right_dims[1]) // 2)
    left_cropped = tf.keras.layers.Cropping2D(cropping=cropping)(left)
    output = tf.keras.layers.Concatenate()([left_cropped, right])
    return output

def unet(inputs, nb_classes):
    
    down_block_1 = conv_block(inputs, [64, 64], 'down_block_1')
    pool1 = downsample(down_block_1, 'pool1')

    down_block_2 = conv_block(pool1, [128, 128], 'down_block_2')
    pool2 = downsample(down_block_2, 'pool2')

    down_block_3 = conv_block(pool2, [256, 256], 'down_block_3')
    pool3 = downsample(down_block_3, 'pool3')

    down_block_4 = conv_block(pool3, [512, 512], 'down_block_4')
    pool4 = downsample(down_block_4, 'pool4')

    down_block_5 = conv_block(pool4, [1024, 1024], 'bottom_block')

    unpool1 = upsample(down_block_5, 512, 'unpool1')
    merge1 = copy_and_crop(down_block_4, unpool1)
    up_block_1 = conv_block(merge1, [512, 512], 'up_block_1')

    unpool2 = upsample(up_block_1, 256, 'unpool2')
    merge2 = copy_and_crop(down_block_3, unpool2)
    up_block_2 = conv_block(merge2, [256, 256], 'up_block_2')

    unpool3 = upsample(up_block_2, 128, 'unpool3')
    merge3 = copy_and_crop(down_block_2, unpool3)
    up_block_3 = conv_block(merge3, [128, 128], 'up_block_3')

    unpool4 = upsample(up_block_3, 64, 'unpool4')
    merge4 = copy_and_crop(down_block_1, unpool4)
    up_block_4 = conv_block(merge4, [64, 64], 'up_block_4')

    output = tf.keras.layers.Conv2D(nb_classes, (1, 1), activation='linear', name='output')(up_block_4)

    print(inputs)
    print(down_block_1)
    print(pool1)
    print(down_block_2)
    print(pool2)
    print(down_block_3)
    print(pool3)
    print(down_block_4)
    print(pool4)
    print(down_block_5)
    
    print(unpool1)
    print(merge1)
    print(up_block_1)
    print(unpool2)
    print(merge2)
    print(up_block_2)
    print(unpool3)
    print(merge3)
    print(up_block_3)
    print(unpool4)
    print(merge4)
    print(up_block_4)
    print(output)
    input()

    return unpool1

