
from common import get_immediate_subdirectories
from common import get_immediate_files
from common import get_immediate_files_with_label
from common import dataPath_Freilassing_48, dataPath_Freilassing1_All


from PIL import Image


import os
import tensorflow as tf
import numpy as np



def getFileInputList(dataPath, mode):
    filename_queue = []
    folderPaths = get_immediate_subdirectories(dataPath)
    #print(folderPaths)
    for folder in folderPaths:
        filePathsWithLabels = get_immediate_files(folder)
        if mode == "train":
            filename_queue.extend(filePathsWithLabels[:50])
        elif mode == "debug":
            filename_queue.append(filePathsWithLabels[0])
            break
        else: 
            filename_queue.extend(filePathsWithLabels[50:])
    
    if mode=="debug":
        print(filename_queue) 

    return filename_queue


def getDataBatches(dataPath, mode, epochs, batchSize):
   
    filename_list = getFileInputList(dataPath, mode)
    #classIDs_list = getClassesIDs(dataPath)

    # step 2
    toShuffle = False if mode=="debug" else True
    filename_queue = tf.train.string_input_producer(filename_list, num_epochs = epochs, shuffle=toShuffle)

    # step 3: read, decode and resize images
    with tf.name_scope('data-reading'):
        reader = tf.WholeFileReader()
        filename, content = reader.read(filename_queue)
        image = tf.image.decode_png(content, channels=3)
        image = tf.cast(image, tf.float32)

        resized_image = tf.image.resize_images(image, [48, 48])

        tf.reshape(filename, [1])
        sp = tf.string_split([filename], "/")
        classID = sp.values[4]    
        classID = tf.string_to_number(classID, out_type=tf.int32)


    # step 4: Batching
    return tf.train.batch([resized_image, classID], batch_size=batchSize, capacity=2000, allow_smaller_final_batch=True)
    
#Optimized input pipeline with TFRecords
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# for int data type
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

###saving in TFRecords
def writeToTFRecords(dataPath, mode):

    filename_queue = []
    folderPaths = get_immediate_subdirectories(dataPath)
    for folder in folderPaths:
        filePathsWithLabels = get_immediate_files_with_label(folder)
        if mode == "train":
            filename_queue.extend(filePathsWithLabels[:50])
        elif mode == "debug":
            filename_queue.append(filePathsWithLabels[0])
            break
        else:
            filename_queue.extend(filePathsWithLabels[50:])

    #print(filename_queue)

    filename = os.path.join(dataPath, mode + ".tfrecords")
    writer = tf.python_io.TFRecordWriter(filename)

    for path,label in filename_queue:
        image = np.array(Image.open(path))
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'image': _bytes_feature(image.tostring())})) 
        writer.write(example.SerializeToString())
    writer.close()

###generate batch from TFRecords
def getDataBatchesWithTFRecords(dataPath, mode,  epochs, batchSize):

    filename=os.path.join(dataPath, mode + ".tfrecords")
    filename_queue=tf.train.string_input_producer([filename], num_epochs=epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)})

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [48, 48, 3])
    image=tf.cast(image, tf.float32)
    label=tf.cast(features['label'], tf.int32)    

    return tf.train.shuffle_batch([image, label], batch_size=batchSize, capacity=2000, min_after_dequeue=1000)

    





writeToTFRecords(dataPath_Freilassing1_All, "train")
