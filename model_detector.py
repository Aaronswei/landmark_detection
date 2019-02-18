
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import math
import numpy as np
import os.path
import tensorflow as tf
import time
from model_train import deepID
import cv2

def per_image_standardization(image):
  mean_image = image - np.mean(image)
  std_image = np.std(image)
  adjusted_stddev = max(std_image, 1.0 / np.sqrt(image.shape[0] * image.shape[1] * image.shape[2]))

  return mean_image / adjusted_stddev




class Predict():
  def __init__(self):
    self.init_model()

  def init_model(self):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      print('Loading model...')
      # Build a Graph that computes the logits predictions from the
      # inference model.
      with tf.device('/cpu:0'):
        self.deepid = deepID(input_shape=[None, 224, 224, 3], n_filters=[20, 40, 60, 80], filter_sizes=[4, 3, 3, 2], activation=tf.nn.relu, dropout=False)

        tf.get_variable_scope().reuse_variables()

      saver = tf.train.Saver()
    
      self.sess = tf.Session()
      ckpt = tf.train.get_checkpoint_state('models/')
      print(ckpt.model_checkpoint_path)
      if ckpt and ckpt.model_checkpoint_path:

        saver.restore(self.sess, ckpt.model_checkpoint_path)

        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
                (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found')


  def detect_image(self, imgName, shape=[224, 224, 3], is_training=False):
    image = cv2.imread(imgName)
    print(image.shape)
    resize_image = cv2.resize(image, (shape[0], shape[1]))

    std_image = per_image_standardization(resize_image)
    h, w = resize_image.shape[:2]
    #new_image = resize_image.copy()
    new_image = std_image.copy()
    new_image = np.array([new_image])


    points = self.sess.run([self.deepid['pred']], feed_dict={self.deepid['x']: new_image, self.deepid['train']: False, self.deepid['keep_prob']: 0.5})[0][0]
    print(points)

    (bbx_left, bbx_right) = points[0]
    (bbx_top, bbx_bottom) = points[1]




    if 0:
      for (x, y) in points:
        y = int(y * h)
        x = int(x * w)
        cv2.circle(resize_image, (x, y), 1, (0, 255, 255), -1)

      cv2.imwrite("test.jpg", resize_image)
     

if __name__ == '__main__':
    predict = Predict()
    txt_list = open('test_list.txt', 'r').readlines()
    imageFile = txt_list[0].split(' ')[0]
    label = np.array(txt_list[0].split(' ')[1:])
    print(imageFile)
    predict.detect_image(imageFile)

