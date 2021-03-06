"""Convolutional neural network for face alignment.

Copyright Mario S. Lew, Oct 2016
"""
import tensorflow as tf
import numpy as np
import os
from libs.tfpipeline import input_pipeline
from libs.batch_norm import batch_norm
from libs import utils
from numpy.linalg import norm
import h5py
import matplotlib.pyplot as plt


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


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

def evaluateError(landmarkGt, landmarkP):
    e = np.zeros(70)
    ocular_dist = norm(landmarkGt[1] - landmarkGt[0])
    for i in range(70):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / ocular_dist
    return e

def evaluateBatchError(landmarkGt, landmarkP, batch_size):
    e = np.zeros([batch_size, 70])
    for i in range(batch_size):
        e[i] = evaluateError(landmarkGt[i], landmarkP[i])
    mean_err = e.mean(axis=0)
    return mean_err

def train_deepid(input_shape=[None, 224, 224, 3],
                n_filters=[20, 40, 60, 80],
                filter_sizes=[4, 3, 3, 2],
                batch_size=64):

    init_learning_rate = 0.005
    decay_rate = 0.95
    decay_steps = 150000

    batch_x, label_x = input_pipeline(['train_list.txt'], batch_size=batch_size, shape=[224, 224, 3], is_training=True)
    
    deepid = deepID(input_shape=input_shape, n_filters=n_filters, filter_sizes=filter_sizes, activation=tf.nn.relu,dropout=False)

    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step * batch_size, decay_steps, decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(deepid['cost'], global_step=global_step)
    saver = tf.train.Saver()
     
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state('models')
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()

        # Ensure no more changes to graph
        tf.get_default_graph().finalize()

        # Start up the queues for handling the image pipeline
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1000000):

            batch_xs, batch_label = sess.run([batch_x, label_x])

            train_cost, pred = sess.run([deepid['cost'], deepid['pred'], optimizer], feed_dict={
                deepid['x']: batch_xs, deepid['y']: batch_label, deepid['train']: True,
                deepid['keep_prob']: 0.5})[:2]
            if i % 100 == 0:
                print(i, train_cost)
                lr = sess.run(learning_rate)
                print('lr: %.10f' % lr)

                batch_label = batch_label.reshape([-1,70,2])
                print('label: ' + np.array_str(batch_label[0]))
                print('pred:  ' + np.array_str(pred[0]))

                err = evaluateBatchError(batch_label, pred, batch_size)
                print('Mean error:' + np.array_str(err))

            if i % 1000 == 0:
                # Save the variables to disk.
                saver.save(sess, "./models/" + 'deepid.ckpt',
                           global_step=i,
                           write_meta_graph=False)
        
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train_deepid()




