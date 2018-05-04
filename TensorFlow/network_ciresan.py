
#ciresan network gtsrb classification
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
    #image input 48 x 48 x 3
    #weights for the first convolution layer

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([7, 7, 3, 100])
        b_conv1 = bias_variable([100])
        #structure of the first convolution layer: convolution, activation, pooling
        h_conv1 = tf.nn.tanh(tf.nn.bias_add(conv2d(inputData, W_conv1), b_conv1))

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    #image input 21 x 21 x 100
    #weights for the second convolution layer 

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([4, 4, 100, 150])
        b_conv2 = bias_variable([150])
        #structure of the second convolution layer: convolution, activation, pooling
        h_conv2 = tf.nn.tanh(tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2))
    
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    #image input 9 x 9 x 150
    #weights for the third convolution layer
    with tf.name_scope('conv3'):        
        W_conv3 = weight_variable([4, 4, 150, 250])
        b_conv3 = bias_variable([250])
        #structure of the second convolution layer: convolution, activation, pooling
        h_conv3 = tf.nn.tanh(tf.nn.bias_add(conv2d(h_pool2, W_conv3), b_conv3))

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    #image input 3 X 3 X 250		
    
    with tf.name_scope('reshape'):
        #conv_nchw = tf.transpose(h_pool3, [0, 3, 1, 2])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 3 * 3 * 250])


    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([3 * 3 * 250, 500])
        b_fc1 = bias_variable([500])
        y_re1 = tf.matmul(h_pool3_flat, W_fc1)
        fc_sum1 = tf.nn.bias_add(y_re1, b_fc1)
        fc_output1 = tf.nn.tanh(fc_sum1) 

    #fully connected layer
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([500, 330])
        b_fc2 = bias_variable([330])
        y_re2 = tf.matmul(fc_output1, W_fc2)
        fc_output2 = tf.nn.bias_add(y_re2, b_fc2)
        
    with tf.name_scope('output'):
        logits = tf.nn.softmax(fc_output2)


    return logits, 330


