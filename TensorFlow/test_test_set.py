import tensorflow as tf

from data_loading import getDataBatchesWithTFRecords
from common import getTrainingPath, getImageSize, getTrainingModel

batchSize = int(sys.argv[1])  #128 for alexnet, 64 for ciresan TODO: to generate dynamically
modelName = str(sys.argv[2])  #actually supported alexnet and ciresan
dataPath = getTrainingPath(modelName)
imSize = getImageSize(modelName)
modelCheckPoint = "Model.ckpt." + str(modelName) + "." +  str(batchSize)

#dynamically import network depending on the input parameter
module = getTrainingModel(modelName)
network_import = "from " + module + " import forward_pass"
exec(network_import)

train_batch, label_batch = getDataBatchesWithTFRecords(dataPath, "test", 1, batchSize, imSize)

output,noClasses = forward_pass(train_batch)
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
saver.restore(sess, modelCheckPoint)

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

