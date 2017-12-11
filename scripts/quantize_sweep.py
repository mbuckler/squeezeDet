########################################################################
# Quantization Sweep
#
# Script for sweeping over the various quantization parameters
#
# Author: Mark Buckler

import subprocess

rounding_methods = ['nearest_neighbor','stochastic']
model_bits_array = [4,5,6,7,8,9,10,11,12]

def run_sweep(gpu,
              rounding_method_idx, 
              reserve_zero_val,
              separate_layer_scales):
    mAPs = []
    for model_bits in model_bits_array:
        rounding_method       = rounding_methods[rounding_method_idx]

        # Define log file name
        log_filename = './log_' + \
                       str(rounding_method_idx) + '_' + \
                       str(reserve_zero_val) + '_' + \
                       str(separate_layer_scales) + '_' + \
                       str(model_bits) + '.txt'

        # Run the evaluation command
        f = open(log_filename,'w')
        subprocess.call('python ./src/eval.py '+
            '--dataset=KITTI '+
            '--data_path=./data/KITTI '+
            '--image_set=val '+
            '--eval_dir=eval_logs_plus '+
            '--run_once=True '+
            '--checkpoint_path=data/model_checkpoints/squeezeDetPlus/model.ckpt-95000 '+
            '--net=squeezeDet+ '+
            ('--gpu='+str(gpu))+' '+
            '--use_quantization=True '+
            ('--rounding_method='+rounding_method)+' '+
            ('--reserve_zero_val='+str(reserve_zero_val))+' '+
            ('--separate_layer_scales='+str(reserve_zero_val))+' '+
            ('--model_bits='+str(model_bits))
            ,stdout=f,stderr=f,shell=True)
        f.close()

        # Parse the log file
        f = open(log_filename, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            if 'Mean average precision:' in line:
                mAP = float(line[-5:])
                mAPs.append(mAP)

    print('Last log:')
    print(log_filename)
    print('Number of model bits per iteration:')
    print(model_bits_array)
    print('Mean Average Precision:')
    print(mAPs)

# Parameters for running the sweep
gpu                   = 0
rounding_method_idx   = 0

reserve_zero_val      = False
separate_layer_scales = False
run_sweep(gpu,
          rounding_method_idx, 
          reserve_zero_val,
          separate_layer_scales)
reserve_zero_val      = False
separate_layer_scales = True
run_sweep(gpu,
          rounding_method_idx, 
          reserve_zero_val,
          separate_layer_scales)
reserve_zero_val      = True
separate_layer_scales = False
run_sweep(gpu,
          rounding_method_idx, 
          reserve_zero_val,
          separate_layer_scales)
reserve_zero_val      = True
separate_layer_scales = True
run_sweep(gpu,
          rounding_method_idx, 
          reserve_zero_val,
          separate_layer_scales)
