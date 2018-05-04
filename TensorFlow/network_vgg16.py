
#ciresan network gtsrb classification
#builds convolutional network graph

import tensorflow as tf
from tensorflow.contrib import slim

def forward_pass(train_batch, dropout):

    with tf.name_scope('input'):
        inputData = tf.identity(train_batch)


    with slim.arg_scope([slim.conv2d, slim.fully_connected],
            activation_fn = tf.nn.relu,
            weights_initializer = tf.truncated_normal_initializer(0.0, 0.01),
            weights_regularizer = slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputData, 2, slim.conv2d, 64, [3, 3], scope='con1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='con2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='con3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='con4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='con5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        net = tf.reshape(net, [-1, 7 * 7 * 512])
        
        net = slim.fully_connected(net, 4096, scope='fc6')
        #todo
        net = slim.dropout(net, dropout, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        #todo
        net = slim.dropout(net, dropout, scope='dropout7')
        net = slim.fully_connected(net, 329, activation_fn=None, scope='fc8')

    with tf.name_scope('output'):
        logits = tf.nn.softmax(net)


    return logits, 329
