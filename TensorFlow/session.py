#run session

import sys
import tensorflow as tf
from data_loading import getDataBatches
from data_loading import getDataBatchesWithTFRecords
from network_ciresan import forward_pass
from common import dataPath_Freilassing_48, dataPath_Freilassing1_All


#input data
batchSize = int(sys.argv[1])
train_batch, label_batch = getDataBatchesWithTFRecords(dataPath_Freilassing1_All,"train", 1000, batchSize)

logits,noClasses = forward_pass(train_batch)

with tf.name_scope('one-hot'):
    label_batch_vector = tf.one_hot(label_batch, noClasses)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_sum(
        tf.losses.log_loss(labels=label_batch_vector, predictions=logits))

with tf.name_scope('adam-optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('counter'):
    counter = tf.Variable(0, trainable=False, dtype=tf.int32)
    inc_counter = tf.assign_add(counter, batchSize)
    print_counter = tf.Print(inc_counter, [counter])


saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

#session preparation
#variable initialization
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# Create a session for running operations in the Graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
# for tensorboard
writer = tf.summary.FileWriter("/tmp/tensorflow/", sess.graph)

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        sess.run(train_step)
        sess.run(print_counter)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()
    saver.save(sess, "./Model.ckpt."+str(batchSize))
    tf.train.write_graph(sess.graph.as_graph_def(), "", "graph"+str(batchSize)+".pb")

# Wait for threads to finish.
coord.join(threads)

sess.close()

