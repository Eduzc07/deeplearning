
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

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='VALID')



def forward_pass(train_batch):

    with tf.name_scope('input'):
        inputData = tf.identity(train_batch)

    #network definition
    #image input 227 x 227 x 3
    #weights for the first convolution layer

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([11, 11, 3, 96])
        b_conv1 = bias_variable([96])
        mul_conv1 = tf.nn.conv2d(inputData, W_conv1, strides = [1, 4, 4, 1], padding = 'SAME')
        h_conv1 = tf.nn.relu(tf.nn.bias_add(mul_conv1, b_conv1))
        
    with tf.name_scope('lrn1'):
        lrn1 = tf.nn.local_response_normalization(h_conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)


    with tf.name_scope('pool1'):
        h_pool1 = max_pool_3x3(lrn1)

    #image input 27 x 27 x 96
    #weights for the second convolution layer 

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 96, 256])
        b_conv2 = bias_variable([256])
        #structure of the second convolution layer: convolution, activation, pooling
        mul_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides = [1, 1, 1, 1], padding = 'SAME')
        h_conv2 = tf.nn.relu(tf.nn.bias_add(mul_conv2, b_conv2))
    

    with tf.name_scope('lrn2'):
        lrn2 = tf.nn.local_response_normalization(h_conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_3x3(lrn2)

    #image input 13 x 13 x 256
    #weights for the third convolution layer
    with tf.name_scope('conv3'):        
        W_conv3 = weight_variable([3, 3, 256, 384])
        b_conv3 = bias_variable([384])
        mul_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides = [1, 1, 1, 1], padding = 'SAME')
        #structure of the second convolution layer: convolution, activation, pooling
        h_conv3 = tf.nn.relu(tf.nn.bias_add(mul_conv3, b_conv3))


    with tf.name_scope('conv4'):        
        W_conv4 = weight_variable([3, 3, 384, 384])
        b_conv4 = bias_variable([384])
        #structure of the second convolution layer: convolution, activation, pooling
        mul_conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides = [1, 1, 1, 1], padding = 'SAME')
        h_conv4 = tf.nn.relu(tf.nn.bias_add(mul_conv4, b_conv4))

    with tf.name_scope('conv5'):        
        W_conv5 = weight_variable([3, 3, 384, 256])
        b_conv5 = bias_variable([256])
        mul_conv5 = tf.nn.conv2d(h_conv4, W_conv5, strides = [1, 1, 1, 1], padding = 'SAME')
        #structure of the second convolution layer: convolution, activation, pooling
        h_conv5 = tf.nn.relu(tf.nn.bias_add(mul_conv5, b_conv5))

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_3x3(h_conv5)

    #image input 6 X 6 X 256		
    
    with tf.name_scope('reshape'):
        h_pool3_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 256])


    #fully connected layer
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([6 * 6 * 256, 330])
        b_fc1 = bias_variable([330])
        y_re1 = tf.matmul(h_pool3_flat, W_fc1)
        fc_sum1 = tf.nn.bias_add(y_re1, b_fc1)
        fc_output1 = tf.nn.relu(fc_sum1)

    #with tf.name_scope('fc2'):
        #W_fc2 = weight_variable([4096, 4096])
        #b_fc2 = bias_variable([4096])
        #y_re2 = tf.matmul(fc_output1, W_fc2)
        #fc_sum2 = tf.nn.bias_add(y_re2, b_fc2)
        #fc_output2 = tf.nn.relu(fc_sum2)

    #with tf.name_scope('fc3'):
        #W_fc3 = weight_variable([4096, 330])
        #b_fc3 = bias_variable([330])
        #y_re3 = tf.matmul(fc_output2, W_fc3)
        #fc_sum3 = tf.nn.bias_add(y_re3, b_fc3)
        #fc_output3 = tf.nn.relu(fc_sum3)
        
    with tf.name_scope('output'):
        logits = tf.nn.softmax(fc_output1)


    return logits, 330


