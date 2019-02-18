#-*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from libs import utils


def deepID(input_shape=[None, 39, 39, 3],
        n_filters=[20, 40, 60, 80],
        filter_sizes=[4, 3, 3, 2],
        activation=tf.nn.relu,
        dropout=False):
    """DeepID.

    Uses tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Shape of the input to the network. e.g. for MNIST: [None, 784].
    n_filters : list, optional
        Number of filters for each layer.
        If convolutional=True, this refers to the total number of output
        filters to create for each layer, with each layer's number of output
        filters as a list.
        If convolutional=False, then this refers to the total number of neurons
        for each layer in a fully connected network.
    filter_sizes : list, optional
        Only applied when convolutional=True.  This refers to the ksize (height
        and width) of each convolutional layer.
    activation : function, optional
        Activation function to apply to each layer, e.g. tf.nn.relu
    dropout : bool, optional
        Whether or not to apply dropout.  If using dropout, you must feed a
        value for 'keep_prob', as returned in the dictionary.  1.0 means no
        dropout is used.  0.0 means every connection is dropped.  Sensible
        values are between 0.5-0.8.

    Returns
    -------
    model : dict
        {
            'cost': Tensor to optimize.
            'Ws': All weights of the encoder.
            'x': Input Placeholder
            'z': Inner most encoding Tensor (latent features)
            'y': Reconstruction of the Decoder
            'keep_prob': Amount to keep when using Dropout
            'corrupt_prob': Amount to corrupt when using Denoising
            'train': Set to True when training/Applies to Batch Normalization.
        }
    """
    # network input / placeholders for train (bn) and dropout
    x = tf.placeholder(tf.float32, input_shape, 'x')
    y = tf.placeholder(tf.float32, [None, 140], 'y')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # 2d -> 4d if convolution
    x_tensor = utils.to_tensor(x)
    current_input = x_tensor

    Ws = []
    shapes = []

    # Build the encoder
    shapes.append(current_input.get_shape().as_list())
    conv1, W = utils.conv2d(x=x_tensor,
                        n_output=n_filters[0],
                        k_h=filter_sizes[0],
                        k_w=filter_sizes[0],
                        d_w=1,
                        d_h=1,
                        name='conv1')
    Ws.append(W)
    # conv1 = activation(batch_norm(conv1, phase_train, 'bn1'))
    conv1 = activation(conv1)


    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    conv2, W = utils.conv2d(x=pool1,
                        n_output=n_filters[1],
                        k_h=filter_sizes[1],
                        k_w=filter_sizes[1],
                        d_w=1,
                        d_h=1,
                        name='conv2')
    Ws.append(W)
    # conv2 = activation(batch_norm(conv2, phase_train, 'bn2'))
    conv2 = activation(conv2)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    conv3, W = utils.conv2d(x=pool2,
                        n_output=n_filters[2],
                        k_h=filter_sizes[2],
                        k_w=filter_sizes[2],
                        d_w=1,
                        d_h=1,
                        name='conv3')
    Ws.append(W)
    # conv3 = activation(batch_norm(conv3, phase_train, 'bn3'))
    conv3 = activation(conv3)

    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    conv4, W = utils.conv2d(x=pool3,
                        n_output=n_filters[3],
                        k_h=filter_sizes[3],
                        k_w=filter_sizes[3],
                        d_w=1,
                        d_h=1,
                        name='conv4')
    Ws.append(W)
    # conv4 = activation(batch_norm(conv4, phase_train, 'bn4'))
    conv4 = activation(conv4)

    pool3_flat = utils.flatten(pool3)
    conv4_flat = utils.flatten(conv4)
  #  concat = tf.concat(1, [pool3_flat, conv4_flat], name='concat')
    concat = tf.concat([pool3_flat, conv4_flat], 1, name='concat')

    ip1, W = utils.linear(concat, 120, name='ip1')
    Ws.append(W)
    ip1 = activation(ip1)
    if dropout:
        ip1 = tf.nn.dropout(ip1, keep_prob)

    ip2, W = utils.linear(ip1, 140, name='ip2')
    Ws.append(W)
    # ip2 = activation(ip2)

    p_flat = utils.flatten(ip2)
    y_flat = utils.flatten(y)

    regularizers = 5e-4 *(tf.nn.l2_loss(Ws[-1]) + tf.nn.l2_loss(Ws[-2]))
    # l2 loss
    loss_x = tf.reduce_sum(tf.squared_difference(p_flat, y_flat), 1)
    cost = tf.reduce_mean(loss_x) + regularizers
    prediction = tf.reshape(p_flat, (-1, 70, 2))

    return {'cost': cost, 
            'Ws': Ws,
            'x': x, 
            'y': y, 
            'pred': prediction,
            'keep_prob': keep_prob,
            'train': phase_train}