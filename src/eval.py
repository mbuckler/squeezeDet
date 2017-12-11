# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
# Quantization support: Mark Buckler (mab598@cornell.edu)

"""Evaluation"""

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

# Imports for parameter manipulation
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import ops
import random

FLAGS = tf.app.flags.FLAGS

# Quantization settings
tf.app.flags.DEFINE_boolean('use_quantization', False, 
                            """Use quantization simulation""")
tf.app.flags.DEFINE_integer('model_bits', 0, 
                            """Number of model bits""")
#tf.app.flags.DEFINE_integer('activation_bits', 0,
#                            """Number of activation bits""")
tf.app.flags.DEFINE_string('rounding_method', 'none',
                            """nearest_neighbor or stochastic""")
tf.app.flags.DEFINE_boolean('reserve_zero_val', False,
                            """Should a value be reserved for true zero""")
tf.app.flags.DEFINE_boolean('separate_layer_scales', False,
                            """Set fixed point scales per layer""")

# General settings
tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently support PASCAL_VOC or KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'test',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for VOC data""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/bichen/logs/squeezeDet/eval',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/bichen/logs/squeezeDet/train',
                            """Path to the training checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                             """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

# Function to generate array of quantized values 
def get_quant_val_array_from_minmax(min_, 
                                    max_, 
                                    num_bits, 
                                    reserve_zero_val):
   
    # Number of representational values
    num_vals = 2**num_bits
    # Amount to step between each representational value
    val_step = float(max_ - min_) / float(num_vals - 1)
    
    # Build the quantized value array
    quant_val_arr = np.empty([num_vals])
    for idx in range(0,num_vals):
        quant_val_arr[idx] = min_ + (float(idx)*val_step)

    # If we must reserve a specific value for 0 then set the closest value to 0
    if reserve_zero_val:
        closest_idx_to_0 = (np.abs(quant_val_arr-0.0)).argmin()
        quant_val_arr[closest_idx_to_0] = 0.0

    return quant_val_arr

# Function to round to quantized value
def round_to_quant_val(quant_val_arr, 
                       in_val, 
                       rounding_method):

    closest_idx = (np.abs(quant_val_arr-in_val)).argmin()

    if rounding_method == 'nearest_neighbor':
        return quant_val_arr[closest_idx]

    elif rounding_method == 'stochastic':
        # If index is at an extreme
        if ((closest_idx == 0) or \
                (closest_idx == (len(quant_val_arr)-1))):
            return quant_val_arr[closest_idx]
        # If the nearest neighbor is below the real value
        elif (in_val > quant_val_arr[closest_idx]):
            # Using > below is counter intuitive. We want the bias to be towards
            # the closest value.
            if (random.uniform(quant_val_arr[closest_idx], \
                    quant_val_arr[closest_idx+1]) > in_val):
                return quant_val_arr[closest_idx]
            else:
                return quant_val_arr[closest_idx+1]
        # If the nearest neighbor is above the real value
        elif (in_val < quant_val_arr[closest_idx]):
            # Using > below is counter intuitive. We want the bias to be towards
            # the closest value.
            if (random.uniform(quant_val_arr[closest_idx-1], \
                    quant_val_arr[closest_idx]) > in_val):
                return quant_val_arr[closest_idx-1]
            else:
                return quant_val_arr[closest_idx]
        # If the nearest neighbor is equal to the real value
        else:
            return quant_val_arr[closest_idx]

    else:
        print('Must specify nearest_neighbor or stochastic rounding')
        exit()
        
def eval_once(
    saver, ckpt_path, summary_writer, eval_summary_ops, eval_summary_phs, imdb,
    model):

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # Restores from checkpoint
    saver.restore(sess, ckpt_path)

    # If we are applying simulated quantization
    if FLAGS.use_quantization:

        # Assertions for validity of quantization arguments
        assert FLAGS.rounding_method != 'none', \
                "Must specify rounding method (nearest_neighbor or stochastic)"
        assert FLAGS.model_bits != 0, \
                "Must specify non-zero number of model bits"
        #assert FLAGS.activation_bits != 0, \
        #        "Must specify non-zero number of activation bits"

        # Extract parameter references for quantization
        all_vars = ops.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)

        # Get global minimums and maximums for weights (kernels) and biases
        global_min = float('inf')
        global_max = 0.0
        global_weight_min = float('inf')
        global_weight_max = 0.0
        global_bias_min = float('inf')
        global_bias_max = 0.0

        global_1x1_min = float('inf')
        global_1x1_max = 0.0
        global_3x3_min = float('inf')
        global_3x3_max = 0.0
        global_conv_min = float('inf')
        global_conv_max = 0.0

        global_1x1_weight_min = float('inf')
        global_1x1_weight_max = 0.0
        global_3x3_weight_min = float('inf')
        global_3x3_weight_max = 0.0
        global_conv_weight_min = float('inf')
        global_conv_weight_max = 0.0

        global_1x1_bias_min = float('inf')
        global_1x1_bias_max = 0.0
        global_3x3_bias_min = float('inf')
        global_3x3_bias_max = 0.0
        global_conv_bias_min = float('inf')
        global_conv_bias_max = 0.0


        for i in range(len(all_vars)):
            print(all_vars[i].name)
            tensor     = sess.run(all_vars[i])
            tensor_min = np.amin(tensor)
            tensor_max = np.amax(tensor)
            # Update global range
            if tensor_min < global_min:
                global_min = tensor_min
                #print('new_min: '+str(global_min))
            if tensor_max > global_max:
                global_max = tensor_max
            # Update kernel and bias ranges
            if ('kernels' in all_vars[i].name):
                if tensor_min < global_weight_min:
                    global_weight_min = tensor_min
                    #print('new_kernel_min: '+str(global_weight_min))
                if tensor_max > global_weight_max:
                    global_weight_max = tensor_max
                # Update 1x1 and 3x3 ranges
                if ('1x1' in all_vars[i].name):
                    if tensor_min < global_1x1_weight_min:
                        global_1x1_weight_min = tensor_min
                        #print('new_1x1_kernel_min: '+str(global_1x1_weight_min))
                    if tensor_max > global_1x1_weight_max:
                        global_1x1_weight_max = tensor_max
                if ('3x3' in all_vars[i].name):
                    if tensor_min < global_3x3_weight_min:
                        global_3x3_weight_min = tensor_min
                        #print('new_3x3_kernel_min: '+str(global_3x3_weight_min))
                    if tensor_max > global_3x3_weight_max:
                        global_3x3_weight_max = tensor_max
                if ('conv' in all_vars[i].name):
                    if tensor_min < global_conv_weight_min:
                        global_conv_weight_min = tensor_min
                        #print('new_conv_kernel_min: '+str(global_conv_weight_min))
                    if tensor_max > global_conv_weight_max:
                        global_conv_weight_max = tensor_max

            if ('biases' in all_vars[i].name):
                if tensor_min < global_bias_min:
                    global_bias_min = tensor_min
                    #print('new_bias_min: '+str(global_bias_min))
                if tensor_max > global_bias_max:
                    global_bias_max = tensor_max
                # Update 1x1 and 3x3 ranges
                if ('1x1' in all_vars[i].name):
                    if tensor_min < global_1x1_bias_min:
                        global_1x1_bias_min = tensor_min
                        #print('new_1x1_bias_min: '+str(global_1x1_bias_min))
                    if tensor_max > global_1x1_bias_max:
                        global_1x1_bias_max = tensor_max
                if ('3x3' in all_vars[i].name):
                    if tensor_min < global_3x3_bias_min:
                        global_3x3_bias_min = tensor_min
                        #print('new_3x3_bias_min: '+str(global_3x3_bias_min))
                    if tensor_max > global_3x3_bias_max:
                        global_3x3_bias_max = tensor_max
                if ('conv' in all_vars[i].name):
                    if tensor_min < global_conv_bias_min:
                        global_conv_bias_min = tensor_min
                        #print('new_conv_bias_min: '+str(global_conv_bias_min))
                    if tensor_max > global_conv_bias_max:
                        global_conv_bias_max = tensor_max

            # Update 1x1, 3x3, and conv ranges
            if ('1x1' in all_vars[i].name):
                if tensor_min < global_1x1_min:
                    global_1x1_min = tensor_min
                if tensor_max > global_1x1_max:
                    global_1x1_max = tensor_max
            if ('3x3' in all_vars[i].name):
                if tensor_min < global_3x3_min:
                    global_3x3_min = tensor_min
                if tensor_max > global_3x3_max:
                    global_3x3_max = tensor_max
            if ('conv' in all_vars[i].name):
                if tensor_min < global_conv_min:
                    global_conv_min = tensor_min
                if tensor_max > global_conv_max:
                    global_conv_max = tensor_max


        print('---')
        print('global spread:')
        print(global_max, global_min)
        print('global weight spread:')
        print(global_weight_max, global_weight_min )
        print('global bias spread:')
        print(global_bias_max,global_bias_min )
        print('---')
        print('global 1x1 spread:')
        print(global_1x1_max, global_1x1_min )
        print('global 1x1 weight spread:')
        print(global_1x1_weight_max, global_1x1_weight_min )
        print('global 1x1 bias spread:')
        print(global_1x1_bias_max, global_1x1_bias_min )
        print('---')
        print('global 3x3 spread:')
        print(global_3x3_max, global_3x3_min )
        print('global 3x3 weight spread:')
        print(global_3x3_weight_max, global_3x3_weight_min )
        print('global 3x3 bias spread:')
        print(global_3x3_bias_max, global_3x3_bias_min )
        print('---')
        print('global conv spread:')
        print(global_conv_max, global_conv_min )
        print('global conv weight spread:')
        print(global_conv_weight_max, global_conv_weight_min )
        print('global conv bias spread:')
        print(global_conv_bias_max, global_conv_bias_min )


        # For each set of parameters
        for i in range(len(all_vars)):
            print(all_vars[i].name)

            # Load the data into a numpy array for easy manipulation
            tensor  = sess.run(all_vars[i])

            # If conv and fire layers are to be scaled separately
            if FLAGS.separate_layer_scales:
                if ('conv' in all_vars[i].name):
                    min_quant_val = global_conv_min
                    max_quant_val = global_conv_max
                elif ('fire' in all_vars[i].name):
                    min_quant_val = min(global_1x1_min,global_3x3_min)
                    max_quant_val = max(global_1x1_max,global_3x3_max)
                else:
                    print("Error: Only conv, 3x3, and 1x1 currently supported")
                    exit()
            else:
                min_quant_val = global_min
                max_quant_val = global_max

            # Get the set of values for quantization
            quant_val_arr = \
                get_quant_val_array_from_minmax(min_quant_val,
                                                max_quant_val,
                                                FLAGS.model_bits,
                                                FLAGS.reserve_zero_val)

            
            # Loop over the whole tensor
            if 'biases' in all_vars[i].name:
                for idx0 in range(0,tensor.shape[0]):
                    tensor[idx0] = round_to_quant_val( \
                                            quant_val_arr,
                                            tensor[idx0],
                                            FLAGS.rounding_method)
            if 'kernels' in all_vars[i].name:
                for idx0 in range(0,tensor.shape[0]):
                  for idx1 in range(0,tensor.shape[1]):
                    for idx2 in range(0,tensor.shape[2]):
                      for idx3 in range(0,tensor.shape[3]):
                        #print('----')
                        #print(tensor[idx0][idx1][idx2][idx3])
                        tensor[idx0][idx1][idx2][idx3] = \
                                round_to_quant_val( \
                                            quant_val_arr,
                                            tensor[idx0][idx1][idx2][idx3],
                                            FLAGS.rounding_method)
                        #print(tensor[idx0][idx1][idx2][idx3])

            # Store the data back into the tensorflow variable
            test_op = tf.assign(all_vars[i], tensor)
            sess.run(test_op)


        '''
        for i in range(len(all_vars)):
            if (('kernels' in all_vars[i].name) and \
                    (not ('Momentum' in all_vars[i].name))):
                if True:
                    test_op = tf.assign(all_vars[i], \
                            tf.scalar_mul(0.90,
                            (all_vars[i])))
                    sess.run(test_op)
                    sess.run(all_vars[i])
        '''


    # Assuming model_checkpoint_path looks something like:
    #   /ckpt_dir/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt_path.split('/')[-1].split('-')[-1]

    num_images = len(imdb.image_idx)

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    _t = {'im_detect': Timer(), 'im_read': Timer(), 'misc': Timer()}

    num_detection = 0.0
    for i in xrange(num_images):
      _t['im_read'].tic()
      images, scales = imdb.read_image_batch(shuffle=False)
      _t['im_read'].toc()

      _t['im_detect'].tic()
      det_boxes, det_probs, det_class = sess.run(
          [model.det_boxes, model.det_probs, model.det_class],
          feed_dict={model.image_input:images})
      _t['im_detect'].toc()

      _t['misc'].tic()
      for j in range(len(det_boxes)): # batch
        # rescale
        det_boxes[j, :, 0::2] /= scales[j][0]
        det_boxes[j, :, 1::2] /= scales[j][1]

        det_bbox, score, det_class = model.filter_prediction(
            det_boxes[j], det_probs[j], det_class[j])

        num_detection += len(det_bbox)
        for c, b, s in zip(det_class, det_bbox, score):
          all_boxes[c][i].append(bbox_transform(b) + [s])
      _t['misc'].toc()

      print ('im_detect: {:d}/{:d} im_read: {:.3f}s '
             'detect: {:.3f}s misc: {:.3f}s'.format(
                i+1, num_images, _t['im_read'].average_time,
                _t['im_detect'].average_time, _t['misc'].average_time))

    print ('Evaluating detections...')
    aps, ap_names = imdb.evaluate_detections(
        FLAGS.eval_dir, global_step, all_boxes)

    print ('Evaluation summary:')
    print ('  Average number of detections per image: {}:'.format(
      num_detection/num_images))
    print ('  Timing:')
    print ('    im_read: {:.3f}s detect: {:.3f}s misc: {:.3f}s'.format(
      _t['im_read'].average_time, _t['im_detect'].average_time,
      _t['misc'].average_time))
    print ('  Average precisions:')

    feed_dict = {}
    for cls, ap in zip(ap_names, aps):
      feed_dict[eval_summary_phs['APs/'+cls]] = ap
      print ('    {}: {:.3f}'.format(cls, ap))

    print ('    Mean average precision: {:.3f}'.format(np.mean(aps)))
    feed_dict[eval_summary_phs['APs/mAP']] = np.mean(aps)
    feed_dict[eval_summary_phs['timing/im_detect']] = \
        _t['im_detect'].average_time
    feed_dict[eval_summary_phs['timing/im_read']] = \
        _t['im_read'].average_time
    feed_dict[eval_summary_phs['timing/post_proc']] = \
        _t['misc'].average_time
    feed_dict[eval_summary_phs['num_det_per_image']] = \
        num_detection/num_images

    print ('Analyzing detections...')
    stats, ims = imdb.do_detection_analysis_in_eval(
        FLAGS.eval_dir, global_step)

    eval_summary_str = sess.run(eval_summary_ops, feed_dict=feed_dict)
    for sum_str in eval_summary_str:
      summary_writer.add_summary(sum_str, global_step)

def evaluate():
  """Evaluate."""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only supports KITTI dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

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

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    # add summary ops and placeholders
    ap_names = []
    for cls in imdb.classes:
      ap_names.append(cls+'_easy')
      ap_names.append(cls+'_medium')
      ap_names.append(cls+'_hard')

    eval_summary_ops = []
    eval_summary_phs = {}
    for ap_name in ap_names:
      ph = tf.placeholder(tf.float32)
      eval_summary_phs['APs/'+ap_name] = ph
      eval_summary_ops.append(tf.summary.scalar('APs/'+ap_name, ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['APs/mAP'] = ph
    eval_summary_ops.append(tf.summary.scalar('APs/mAP', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/im_detect'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/im_detect', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/im_read'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/im_read', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['timing/post_proc'] = ph
    eval_summary_ops.append(tf.summary.scalar('timing/post_proc', ph))

    ph = tf.placeholder(tf.float32)
    eval_summary_phs['num_det_per_image'] = ph
    eval_summary_ops.append(tf.summary.scalar('num_det_per_image', ph))

    saver = tf.train.Saver(model.model_params)

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    
    ckpts = set() 
    while True:
      if FLAGS.run_once:
        # When run_once is true, checkpoint_path should point to the exact
        # checkpoint file.
        eval_once(
            saver, FLAGS.checkpoint_path, summary_writer, eval_summary_ops,
            eval_summary_phs, imdb, model)
        return
      else:
        # When run_once is false, checkpoint_path should point to the directory
        # that stores checkpoint files.
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
          if ckpt.model_checkpoint_path in ckpts:
            # Do not evaluate on the same checkpoint
            print ('Wait {:d}s for new checkpoints to be saved ... '
                      .format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)
          else:
            ckpts.add(ckpt.model_checkpoint_path)
            print ('Evaluating {}...'.format(ckpt.model_checkpoint_path))
            eval_once(
                saver, ckpt.model_checkpoint_path, summary_writer,
                eval_summary_ops, eval_summary_phs, imdb, model)
        else:
          print('No checkpoint file found')
          if not FLAGS.run_once:
            print ('Wait {:d}s for new checkpoints to be saved ... '
                      .format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
