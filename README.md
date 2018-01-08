# Gleason U-net Project

The goal of the project is to segment prostate cancer tissues by [gleason scoring system](https://en.wikipedia.org/wiki/Gleason_grading_system).  
Currently, we're using deep learning model - [U-net](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net) architecture.  

## Components

* extract_model/ : extract weight parameters from pre-trained U-net caffe model to npy's
* param/ : training Gleason_U-net weight & bias parameters
* testimages/ : sample test images
* unet.py : Our Gleason_U-net object
* coco.py : training code on [cocodatasets](https://github.com/cocodataset/cocoapi)
* test.py : test out Gleason_Unet and plot
* resnet.py : Resnet-152 classifier for gleason scoring
* init_resnet.py : initializer of resnet weight parameters

## Pre-requisites

* python3
* tensorflow
* numpy
* scikit-image
* matplotlib
* pycocotools (optional)
