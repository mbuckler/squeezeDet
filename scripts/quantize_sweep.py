########################################################################
# Quantization Sweep
#
# Script for sweeping over the various quantization parameters
#
# Author: Mark Buckler

import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rounding_methods = ['nearest_neighbor','stochastic']
model_bits_array = [4,5,6,7,9,11]

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

    return mAPs

# Parameters for running the sweep
sweep_labels = ['nearneighbor_nozero',
                'nearneighbor_zero',
                'stochastic_nozero',
                'stochastic_zero']
mAPs_to_plot = []
gpu                   = 0
separate_layer_scales = False

# Run four sweeps to plot
rounding_method_idx   = 0
reserve_zero_val      = False
mAPs_to_plot.append(run_sweep(gpu,
                    rounding_method_idx,
                    reserve_zero_val,
                    separate_layer_scales))

reserve_zero_val      = True
mAPs_to_plot.append(run_sweep(gpu,
                    rounding_method_idx,
                    reserve_zero_val,
                    separate_layer_scales))


rounding_method_idx   = 1
reserve_zero_val      = False
mAPs_to_plot.append(run_sweep(gpu,
                    rounding_method_idx,
                    reserve_zero_val,
                    separate_layer_scales))

reserve_zero_val      = True
mAPs_to_plot.append(run_sweep(gpu,
                    rounding_method_idx,
                    reserve_zero_val,
                    separate_layer_scales))

# Plot mAP data
plt.plot(model_bits_array,mAPs_to_plot[0],'r',label=sweep_labels[0])
plt.plot(model_bits_array,mAPs_to_plot[1],'k',label=sweep_labels[1])
plt.plot(model_bits_array,mAPs_to_plot[2],'b',label=sweep_labels[2])
plt.plot(model_bits_array,mAPs_to_plot[3],'g',label=sweep_labels[3])

# Set up legend
plt.legend(loc='lower right')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                   ncol=2, mode="expand", borderaxespad=0.)

# Title and axis labels
plt.title('SqueezeDet Accuracy with Low Precision Parameters')
plt.xlabel('Bits per model parameter')
plt.ylabel('Mean Average Precision (mAP)')

# Write the plot to file
plt.savefig('quantization_sweep_plot.png')





