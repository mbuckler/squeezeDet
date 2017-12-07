# Author: Mark Buckler

"""Quantization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import pascal_voc, kitti
from utils.util import bbox_transform, Timer
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/bichen/logs/squeezeDet/train',
                            """Path to the training checkpoint.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture.""")

from tensorflow.python import pywrap_tensorflow
'''
def eval_once(
    saver, ckpt_path, model):

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # Restores from checkpoint
    saver = 
    saver.restore(sess, ckpt_path)

    checkpoint_path = ckpt_path
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key)) 
        test_0 = reader.get_tensor(key)
        assign_op = test_0.assign(test_0 * 0)
        sess.run(assign_op)
        print(reader.get_tensor(key))
        exit()
'''
def quantize():

  os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  with tf.Graph().as_default() as g:

    assert FLAGS.net == 'vgg16' or FLAGS.net == 'resnet50' \
        or FLAGS.net == 'squeezeDet' or FLAGS.net == 'squeezeDet+', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'vgg16':
      mc = kitti_vgg16_config()
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = VGG16ConvDet(mc)
    elif FLAGS.net == 'resnet50':
      mc = kitti_res50_config()
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = ResNet50ConvDet(mc)
    elif FLAGS.net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc)
    elif FLAGS.net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc)

    #####saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_path = FLAGS.checkpoint_path
        saver = tf.train.import_meta_graph(ckpt_path+'.meta')
        saver.restore(sess, ckpt_path)
        exit()
        
    #ckpts = set() 
    #eval_once(FLAGS.checkpoint_path,model)
    return
       

def main(argv=None):  # pylint: disable=unused-argument
  quantize()


if __name__ == '__main__':
  tf.app.run()
