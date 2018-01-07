# Extraction

The extract.py is referred from [extract-caffe-params](https://github.com/nilboy/extract-caffe-params).  
You can get the origin caffemodel file from [here](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-release-2015-10-02.tar.gz).  

## Pre-requisites

* caffe
* numpy
* u-net caffemodel

## Run

To extract caffemodel to numpy array files,  
```shell
python3 extract.py --model phseg_v5-train.prototxt --weights phseg_v5.caffemodel --output ../param/
python3 transpose.py
```
