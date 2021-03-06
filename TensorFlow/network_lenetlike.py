
#network similar to lenet
#builds convolutional network graph

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')



def forward_pass(train_batch):

    with tf.name_scope('input'):
        inputData = tf.identity(train_batch)

    #network definition
    #image input 32 x 32 x 3
    #weights for the first convolution layer

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, 6])
        b_conv1 = bias_variable([6])
        #structure of the first convolution layer: convolution, activation, pooling
        h_conv1 = tf.nn.tanh(tf.nn.bias_add(conv2d(inputData, W_conv1), b_conv1))

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    #image input 14 x 14 x 6
    #weights for the second convolution layer 

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 6, 16])
        b_conv2 = bias_variable([16])
        #structure of the second convolution layer: convolution, activation, pooling
        h_conv2 = tf.nn.tanh(tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2))
    
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    #image input 10 x 10 x 16

	
    
    with tf.name_scope('reshape'):
        conv_nchw = tf.transpose(h_pool2, [0, 3, 1, 2])
        h_pool2_flat = tf.reshape(conv_nchw, [-1, 5 * 5 * 16])


    #fully connected layer
    with tf.name_scope('fc'):
        W_fc1 = weight_variable([5 * 5 * 16, 46])
        b_fc1 = bias_variable([46])
        y_re = tf.matmul(h_pool2_flat, W_fc1)
        fc_output = tf.nn.bias_add(y_re, b_fc1)
        
    with tf.name_scope('output'):
        logits = tf.nn.softmax(fc_output)


    return logits, 46


