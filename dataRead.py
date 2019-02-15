import tensorflow as tf
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import cv2

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


def read_my_file_format(filename):
    

    record_defaults = [[""]] + [[1.0]] * 140
    components = tf.decode_csv(filename, record_defaults=record_defaults, field_delim=" ")
    imgName = components[0]
    features = components[1:]
    img_contents = tf.read_file(imgName)
    img = tf.image.decode_jpeg(img_contents, channels=3)
    return img, features


if __name__ == '__main__':
    shape = [224, 224, 3]

    filename_queue = tf.train.string_input_producer(['train_list.txt'], shuffle=True)
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    img, features = read_my_file_format(value)

    image = tf.image.resize_images(img, [shape[0],shape[1]])
    img_reshape = tf.cast(image, tf.float32)
    float_image = tf.image.per_image_standardization(img_reshape)

    min_after_dequeue = 80000 // 100

    # The capacity should be larger than min_after_dequeue, and determines how
    # many examples are prefetched.  TF docs recommend setting this value to:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + (2 + 1) * 64

    # Randomize the order and output batches of batch_size.
    img_batch, label_batch = tf.train.shuffle_batch([float_image, features],
                                   enqueue_many=False,
                                   batch_size=64,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=2)

    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        im, fea = sess.run([img_batch, label_batch])
        print(im[0].shape)
      #  cv2.imshow("test", im[0])
        coord.request_stop()
        coord.join(threads)