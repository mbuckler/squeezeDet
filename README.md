## _SqueezeDet - Low Precision
Original work by Bichen Wu, Alvin Wan, Forrest Iandola, Peter H. Jin, Kurt Keutzer (UC Berkeley & DeepScale)
Addition of low precision evaluation by [Mark Buckler](http://www.markbuckler.com/).

This repository contains a tensorflow implementation of SqueezeDet, a
convolutional neural network based object detector described in this paper:
https://arxiv.org/abs/1612.01051. If you find this work useful for your
research, please consider citing:

    @inproceedings{squeezedet,
        Author = {Bichen Wu and Forrest Iandola and Peter H. Jin and Kurt Keutzer},
        Title = {SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving},
        Journal = {arXiv:1612.01051},
        Year = {2016}
    }

## Low Precision Implementation

This fork contains additions to the original SqueezeDet project. Specifically,
functionality has been added to simulate evaluation of SqueezeDet models when
using fixed point model parameters. "Simulation" of fixed point model parameters
means that floating point math is still used in computation, but the number of
values which can be used to represent a parameter is artificially limited based
on the number of simulated fixed point bits.

To convert floating point parameters to simulated fixed point parameters first a
given range is set. The default range is given by the maximum and minimum model
parameter values before conversion. Then, valid values are defined as a linear
distribution of 2^N possible values between this range (where N is the number of
simulated fixed point bits). Model conversion consists of rounding each of the
parameters to one of these valid values.

### Supported Conversion Methods

In addition to supporting an arbitrary number of simulated bits the code
supports a few different conversion methods.

- Rounding method:

	Both nearest neighbors and stochastic rounding are supported.

- Zero reservation:

	The default model conversion doesn't reserve a valid value for exact
zero, but it can if requested.

- Per-layer scale:

	Rather than setting the initial scale based on the max and min for the
entire model, it can be chosen per layer.

### Usage

After performing the installation and downloading the KITTI dataset
(instructions for that found below) you can evaluate a given low precision
configuration with the following command:

  ```Shell
  python ./src/eval.py \
    --dataset=KITTI \
    --data_path=./data/KITTI \
    --image_set=val \
    --eval_dir=eval_logs_plus \
    --run_once=True \
    --checkpoint_path=data/model_checkpoints/squeezeDetPlus/model.ckpt-95000 \
    --net=squeezeDet+ \
    --gpu=0 \
	--use_quantization=True \
	--model_bits=10 \
	--rounding_method=stochastic \
	--reserve_zero_val=True \
	--separate_layer_scales=False \
    &> test_log_plus.txt
  ```

To perform a full sweep with different options you can run the provided script
(shown below). This script will write out log files as well as plot results.

  ```Shell
  python ./scripts/quantize_sweep.py
  ```

### Results

The sample output of the sweep script can be found below. Noteably, these tests
don't include any examples with separate layer scales as this wasn't found to
improve model accuracy at all. What does help significantly is stochastic
rounding and zero reservation, and when used together they offer the highest
accuracy per bit.

![sample results](https://github.com/mbuckler/squeezeDet-low-precision/blob/master/data/quantization_sweep_plot_sample.png)

## Installation:

The following instructions are written for Linux-based distros.

- Clone the SqueezeDet repository:

  ```Shell
  git clone https://github.com/BichenWuUCB/squeezeDet.git
  ```
  Let's call the top level directory of SqueezeDet `$SQDT_ROOT`. 

- (Optional) Setup your own virtual environment.

  1. The following assumes `python` is the Python2.7 executable. Navigate to your user home directory, and create the virtual environment there.
  
    ```Shell
    cd ~
    virtualenv env --python=python
    ```
    
  2. Launch the virtual environment.
  
    ```Shell
    source ~/env/bin/activate
    ```
    
- Use pip to install required Python packages:
    
    ```Shell
    pip install -r requirements.txt
    ```
## Demo:
- Download SqueezeDet model parameters from [here](https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz?dl=0), untar it, and put it under `$SQDT_ROOT/data/` If you are using command line, type:

  ```Shell
  cd $SQDT_ROOT/data/
  wget https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz
  tar -xzvf model_checkpoints.tgz
  rm model_checkpoints.tgz
  ```


- Now we can run the demo. To detect the sample image `$SQDT_ROOT/data/sample.png`,

  ```Shell
  cd $SQDT_ROOT/
  python ./src/demo.py
  ```
  If the installation is correct, the detector should generate this image: ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/out_sample.png)

  To detect other image(s), use the flag `--input_path=./data/*.png` to point to input image(s). Input image(s) will be scaled to the resolution of 1242x375 (KITTI image resolution), so it works best when original resolution is close to that.  

- SqueezeDet is a real-time object detector, which can be used to detect videos. The video demo will be released later.

## Training/Validation:
- Install gnuplot from your package manager (needed for validation)

  ```Shell
  sudo apt-get install gnuplot-x11
  ```

- Download KITTI object detection dataset: [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and [labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip). Put them under `$SQDT_ROOT/data/KITTI/`. Unzip them, then you will get two directories:  `$SQDT_ROOT/data/KITTI/training/` and `$SQDT_ROOT/data/KITTI/testing/`. 

- Now we need to split the training data into a training set and a validation set. 

  ```Shell
  cd $SQDT_ROOT/data/KITTI/
  mkdir ImageSets
  cd ./ImageSets
  ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
  ```
  `trainval.txt` contains indices to all the images in the training data. In our experiments, we randomly split half of indices in `trainval.txt` into `train.txt` to form a training set and rest of them into `val.txt` to form a validation set. For your convenience, we provide a script to split the train-val set automatically. Simply run
  
  ```Shell
  cd $SQDT_ROOT/data/
  python random_split_train_val.py
  ```
  
  then you should get the `train.txt` and `val.txt` under `$SQDT_ROOT/data/KITTI/ImageSets`. 

  When above two steps are finished, the structure of `$SQDT_ROOT/data/KITTI/` should at least contain:

  ```Shell
  $SQDT_ROOT/data/KITTI/
                    |->training/
                    |     |-> image_2/00****.png
                    |     L-> label_2/00****.txt
                    |->testing/
                    |     L-> image_2/00****.png
                    L->ImageSets/
                          |-> trainval.txt
                          |-> train.txt
                          L-> val.txt
  ```

- Next, download the CNN model pretrained for ImageNet classification:
  ```Shell
  cd $SQDT_ROOT/data/
  # SqueezeNet
  wget https://www.dropbox.com/s/fzvtkc42hu3xw47/SqueezeNet.tgz
  tar -xzvf SqueezeNet.tgz
  # ResNet50 
  wget https://www.dropbox.com/s/p65lktictdq011t/ResNet.tgz
  tar -xzvf ResNet.tgz
  # VGG16
  wget https://www.dropbox.com/s/zxd72nj012lzrlf/VGG16.tgz
  tar -xzvf VGG16.tgz
  ```

- Now we can start training. Training script can be found in `$SQDT_ROOT/scripts/train.sh`, which contains commands to train 4 models: SqueezeDet, SqueezeDet+, VGG16+ConvDet, ResNet50+ConvDet. 
  ```Shell
  cd $SQDT_ROOT/
  ./scripts/train.sh -net (squeezeDet|squeezeDet+|vgg16|resnet50) -train_dir /tmp/bichen/logs/squeezedet -gpu 0
  ```

  Training logs are saved to the directory specified by `-train_dir`. GPU id is specified by `-gpu`. Network to train is specificed by `-net` 

- Before evaluation, you need to first compile the official evaluation script of KITTI dataset
  ```Shell
  cd $SQDT_ROOT/src/dataset/kitti-eval
  make
  ```

- Then, you can launch the evaluation script (in parallel with training) by 

  ```Shell
  cd $SQDT_ROOT/
  ./scripts/eval.sh -net (squeezeDet|squeezeDet+|vgg16|resnet50) -eval_dir /tmp/bichen/logs/squeezedet -image_set (train|val) -gpu 1
  ```

  Note that `-train_dir` in the training script should be the same as `-eval_dir` in the evaluation script to make it easy for tensorboard to load logs. 

  You can run two evaluation scripts to simultaneously evaluate the model on training and validation set. The training script keeps dumping checkpoint (model parameters) to the training directory once every 1000 steps (step size can be changed). Once a new checkpoint is saved, evaluation threads load the new checkpoint file and evaluate them on training and validation set. 

- Finally, to monitor training and evaluation process, you can use tensorboard by

  ```Shell
  tensorboard --logdir=$LOG_DIR
  ```
  Here, `$LOG_DIR` is the directory where your training and evaluation threads dump log events, which should be the same as `-train_dir` and `-eval_dir` specified in `train.sh` and `eval.sh`. From tensorboard, you should be able to see a lot of information including loss, average precision, error analysis, example detections, model visualization, etc.

  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/detection_analysis.png)
  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/graph.png)
  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/det_img.png)

- If you would like to simply run an evaluation on a given model, use the python
  evaluation script directly

  ```Shell
  python ./src/eval.py \
    --dataset=KITTI \
    --data_path=./data/KITTI \
    --image_set=val \
    --eval_dir=eval_logs_plus \
    --run_once=True \
    --checkpoint_path=data/model_checkpoints/squeezeDetPlus/model.ckpt-95000 \
    --net=squeezeDet+ \
    --gpu=0 \
    &> test_log_plus.txt
  ```

