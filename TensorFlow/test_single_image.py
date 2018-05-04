import tensorflow as tf

from data_loading import getDataBatches
from network_ciresan import forward_pass

dataPath="/storage/cucu/MachineLearning/Freilassing1_All_48X48_png/"
train_batch, label_batch = getDataBatches(dataPath, "debug", 1, 1)
#image_toPrint = tf.transpose(train_batch, [0, 2, 3, 1])

output, noClasses = forward_pass(train_batch)
print_output = tf.Print(output, [output], summarize = 5000)

saver = tf.train.Saver()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
# it is required to somehow start the streaming
sess.run(init_op)

# when I move this befor run(init_op) the resulted accuracy is small
saver.restore(sess, 'Model.ckpt.ciresan.1')

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        # Run training steps or whatever
        #sess.run(print_accuracy)
        sess.run(print_output)
        #sess.run(print_counter1)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)

sess.close()

