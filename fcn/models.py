import tensorflow as tf
import tensornets as nets

def get_vgg16_pooling_layer(layer_num):
    name = "vgg16/conv{}/pool/MaxPool:0".format(layer_num)
    output = tf.get_default_graph().get_tensor_by_name(name)
    return output

def fcn_32(sess, inputs, nb_classes):    
    vgg16_pretrained = nets.VGG16(inputs, is_training=True, stem=True)
    sess.run(vgg16_pretrained.pretrained())
    print(vgg16_pretrained)
    
    conv6 = tf.layers.conv2d(vgg16_pretrained, 4096, (7,7), padding='same', activation='relu', name='conv6')
    print(conv6)
    drop6 = tf.layers.dropout(conv6)
    print(drop6)
    
    conv7 = tf.layers.conv2d(drop6, 4096, (1,1), padding='same', activation='relu', name='conv7')
    print(conv7)
    drop7 = tf.layers.dropout(conv7)
    print(drop7)
    
    conv7_score = tf.layers.conv2d(drop7, nb_classes, (1,1), padding='same', activation='relu', name='conv7_score')
    print(conv7_score)
    
    final_score = tf.layers.conv2d_transpose(conv7_score, nb_classes, (32,32), strides=(32,32), padding='valid', activation=None, name='final_score')
    print(final_score)
    return final_score

def fcn_16(sess, inputs, nb_classes):    
    vgg16_pretrained = nets.VGG16(inputs, is_training=True, stem=True)
    sess.run(vgg16_pretrained.pretrained())
    print(vgg16_pretrained)
    
    conv6 = tf.layers.conv2d(vgg16_pretrained, 4096, (7,7), padding='same', activation='relu', name='conv6')
    print(conv6)
    drop6 = tf.layers.dropout(conv6)
    print(drop6)
    
    conv7 = tf.layers.conv2d(drop6, 4096, (1,1), padding='same', activation='relu', name='conv7')
    print(conv7)
    drop7 = tf.layers.dropout(conv7)
    print(drop7)

    # Get class scores from conv7
    conv7_score = tf.layers.conv2d(drop7, nb_classes, (1,1), padding='same', activation='relu', name='conv7_score')
    print(conv7_score)

    # Upsample conv7    
    conv7_score_x2 = tf.layers.conv2d_transpose(conv7_score, nb_classes, (2,2), strides=(2,2), padding='valid', activation=None, name='conv7_score_x2')
    print(conv7_score_x2)

    # Get pool4 and class scores
    pool4 = get_vgg16_pooling_layer(4)
    print(pool4)
    pool4_score = tf.layers.conv2d(pool4, nb_classes, (1,1), padding='same', activation='relu', name='pool4_score')
    print(pool4_score)

    # Compute sum
    pool4_conv7_score = tf.math.add(pool4_score, conv7_score_x2)

    # Upsample sum
    final_score = tf.layers.conv2d_transpose(pool4_conv7_score, nb_classes, (16,16), strides=(16,16), padding='valid', activation=None, name='final_score')
    print(final_score)
    return final_score

def fcn_8(sess, inputs, nb_classes):
    vgg16_pretrained = nets.VGG16(inputs, is_training=True, stem=True)
    sess.run(vgg16_pretrained.pretrained())
    print(vgg16_pretrained)
    
    conv6 = tf.layers.conv2d(vgg16_pretrained, 4096, (7,7), padding='same', activation='relu', name='conv6')
    print(conv6)
    drop6 = tf.layers.dropout(conv6)
    print(drop6)
    
    conv7 = tf.layers.conv2d(drop6, 4096, (1,1), padding='same', activation='relu', name='conv7')
    print(conv7)
    drop7 = tf.layers.dropout(conv7)
    print(drop7)

    # Get class scores from conv7
    conv7_score = tf.layers.conv2d(drop7, nb_classes, (1,1), padding='same', activation='relu', name='conv7_score')
    print(conv7_score)

    # Upsample conv7    
    conv7_score_x4 = tf.layers.conv2d_transpose(conv7_score, nb_classes, (4,4), strides=(4,4), padding='valid', activation=None, name='conv7_score_x4')
    print(conv7_score_x4)

    # Get pool4 and class scores
    pool4 = get_vgg16_pooling_layer(4)
    print(pool4)
    pool4_score = tf.layers.conv2d(pool4, nb_classes, (1,1), padding='same', activation='relu', name='pool4_score')
    print(pool4_score)
    pool4_score_x2 = tf.layers.conv2d_transpose(pool4_score, nb_classes, (2,2), strides=(2,2), padding='valid', activation=None, name='pool4_score_x2')
    print(pool4_score_x2)

    # Get pool3 and class scores
    pool3 = get_vgg16_pooling_layer(3)
    print(pool3)
    pool3_score = tf.layers.conv2d(pool3, nb_classes, (1,1), padding='same', activation='relu', name='pool3_score')
    print(pool3_score)

    # Compute sum
    sum_score = tf.math.add_n([conv7_score_x4, pool4_score_x2, pool3_score])

    # Upsample sum
    final_score = tf.layers.conv2d_transpose(sum_score, nb_classes, (8,8), strides=(8,8), padding='valid', activation=None, name='final_score')
    print(final_score)
    return final_score
