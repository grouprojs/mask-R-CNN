# Mask R-CNN

## This repo referenced the source code from pytorch/torchvision modules
* https://github.com/pytorch/vision/tree/master/references/detection

## Environment
```
conda create --name cs7643-final pip
conda activate cs7643-final
pip install -r requirements
```

## Files：
```
  ├── backbone: feature extraction network
  ├── network_files: Mask R-CNN network
  ├── train_utils: utilities for training and validation
  ├── my_dataset_coco.py: load coco
  ├── my_dataset_voc.py: load voc
  ├── train.py: script for single GPU training
  ├── train_multi_GPU.py: script for multiple GPU training
  ├── predict.py: predict a test image using trained weights
  ├── validation.py: validate coco using traied weights, create record_mAP.txt
  └── transforms.py: preprocess data 
```

## pre-trained weights (download into ./)
```
wget https://download.pytorch.org/models/resnet50-0676ba61.pth  # resnet-50 pre-trained weights using Imagenet
wget https://download.pytorch.org/models/resnet101-63fe2227.pth # resnet-101 pre-trained weights using Imagenet
wget https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth # maskrcnn resnet-50-fpn pre-trained weights using COCO
```
 
## dataset
### COCO2017
* Download at: https://cocodataset.org/
* Save into ./data/coco2017

### Pascal VOC2012
* Download at： http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
* Save into ./data/VOCdevkit

## Training
* Download training data set
* Download pre-trained weights
* Set `--num-classes` and `--data-path` as they are different for VOC and COCO
* use train.py if training with one GPU
* use `torchrun --nproc_per_node=8 train_multi_GPU.py` if training with multiple GPUs, `nproc_per_node` specifies number of GPUs

## Notes
* set`--data-path` as the root path where data is located. 
* for COCO, save COCO into ./data/coco2017
```
python train.py --data-path /data/coco2017
```
* for VOC, save into ./data/VOCdevkit
```
python train.py --data-path /data/VOCdevkit
```
