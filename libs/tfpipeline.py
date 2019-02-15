import tensorflow as tf
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

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

def processImage(img):
    """
        process images before feeding to CNNs
        imgs: W x H x 1
    """
    img = img.astype(np.float32)
    m = img.mean()
    s = img.std()
    img = (img - m) / s
    return img

def input_pipeline(TXTs, batch_size, shape, is_training=False):
    filename_queue = tf.train.string_input_producer(TXTs, shuffle=is_training)
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    img, features = read_my_file_format(value)


    image = tf.image.resize_images(img, [shape[0], shape[1]])
    img_reshape = tf.cast(image, tf.float32)
    float_image = tf.image.per_image_standardization(img_reshape)
    if is_training:
        float_image = distort_color(float_image)

    min_after_dequeue = 80000 // 100

    # The capacity should be larger than min_after_dequeue, and determines how
    # many examples are prefetched.  TF docs recommend setting this value to:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + (2 + 1) * batch_size

    # Randomize the order and output batches of batch_size.
    img_batch, label_batch = tf.train.shuffle_batch([float_image, features],
                                   enqueue_many=False,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=2)

    return img_batch, label_batch

def distort_color(image, thread_id=0, stddev=0.1, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        image += tf.random_normal(
                tf.shape(image),
                stddev=stddev,
                dtype=tf.float32,
                seed=42,
                name='add_gaussian_noise')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

