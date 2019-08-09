import tensorflow as tf
import tensornets as nets

def get_vgg16_pooling_layer(layer_num):
    name = "vgg16/conv{}/pool/MaxPool:0".format(layer_num)
    output = tf.get_default_graph().get_tensor_by_name(name)
    return output

def conv2d_helper(inputs, filters, kernel_size, name, do_dropout=True):
    x = tf.layers.conv2d(inputs, filters, kernel_size, padding='same', activation='relu', name=name)
    if do_dropout:
        x = tf.layers.dropout(x)
    return x

def upsample_helper(inputs, filters, scale, name):
    x = tf.layers.conv2d_transpose(inputs, filters, (scale,scale), strides=(scale,scale), padding='valid', activation=None, name=name)
    return x

def fcn_32(sess, inputs, nb_classes):    
    vgg16_pretrained = nets.VGG16(inputs, is_training=True, stem=True)
    sess.run(vgg16_pretrained.pretrained())

    # Replace fc layers with conv layers
    conv6 = conv2d_helper(vgg16_pretrained, 4096, (7,7), name='conv6')
    conv7 = conv2d_helper(conv6, 4096, (1,1), name='conv7')
    conv7_score = conv2d_helper(conv7, nb_classes, (1,1), name='conv7_score', do_dropout=False)

    final_score = upsample_helper(conv7_score, nb_classes, 32, name='final_score')
    return final_score

def fcn_16(sess, inputs, nb_classes):    
    vgg16_pretrained = nets.VGG16(inputs, is_training=True, stem=True)
    sess.run(vgg16_pretrained.pretrained())
    
    # Replace fc layers with conv layers
    conv6 = conv2d_helper(vgg16_pretrained, 4096, (7,7), name='conv6')
    conv7 = conv2d_helper(conv6, 4096, (1,1), name='conv7')
    conv7_score = conv2d_helper(conv7, nb_classes, (1,1), name='conv7_score', do_dropout=False)

    # Upsample conv7    
    conv7_score_x2 = upsample_helper(conv7_score, nb_classes, 2, name='conv7_score_x2')

    # Get pool4 and class scores
    pool4 = get_vgg16_pooling_layer(4)
    pool4_score = conv2d_helper(pool4, nb_classes, (1,1), name='pool4_score')

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
