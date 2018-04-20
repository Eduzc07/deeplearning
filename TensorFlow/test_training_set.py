import tensorflow as tf

from data_loading import getDataBatches
from network_ciresan import forward_pass
from common import dataPath_Freilassing_48, dataPath_Freilassing1_All
from data_loading import getDataBatchesWithTFRecords


#train_batch, label_batch = getDataBatches(dataPath_Freilassing1_All,"train", 1,64)

train_batch, label_batch = getDataBatchesWithTFRecords(dataPath_Freilassing1_All,"train", 1, 64)

output, noClasses = forward_pass(train_batch)
label_batch_vector = tf.one_hot(label_batch, noClasses)
_, accuracy = tf.metrics.accuracy(tf.argmax(output, 1), tf.argmax(label_batch_vector, 1))
print_accuracy = tf.Print(accuracy, [accuracy])

saver = tf.train.Saver()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


#session preparation
#variable initialization
#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
# it is required to somehow start the streaming
sess.run(init_op)

# when I move this befor run(init_op) the resulted accuracy is small
saver.restore(sess, 'Model.ckpt.64')

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)



try:
    while not coord.should_stop():
        # Run training steps or whatever
        sess.run(print_accuracy)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)


sess.close()

