#! /usr/bin/env python
# coding=utf-8
# get config by: from config import cfg
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.ARCH                       = edict()
__C.ARCH.CLASSES               = "./yolo/checkpoints/yolov4-416-coco-darknet/coco.names"
__C.ARCH.ANCHORS               = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
__C.ARCH.ANCHORS_V3            = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
__C.ARCH.ANCHORS_TINY          = [23, 27, 37, 58, 81, 82, 81, 82, 135, 169, 344, 319]
__C.ARCH.STRIDES               = [8, 16, 32]
__C.ARCH.STRIDES_TINY          = [16, 32]
__C.ARCH.XYSCALE               = [1.2, 1.1, 1.05]
__C.ARCH.XYSCALE_TINY          = [1.05, 1.05]
__C.ARCH.ANCHOR_PER_SCALE      = 3
__C.ARCH.IOU_LOSS_THRESH       = 0.5

__C.MODEL                      = edict()
__C.MODEL.WEIGHTS_PATH         = 'yolo/checkpoints/yolov4-416-coco-tf'
__C.MODEL.INPUT_SIZE           = 416
__C.MODEL.SCORE_THRESHOLD      = 0.7
__C.MODEL.IOU_THRESHOLD        = 0.45

__C.EVALUATE                   = edict()
__C.EVALUATE.ANNOT_PATH        = "./data/dataset/val2017.txt"
__C.EVALUATE.DECTECTED_IMAGE_PATH = "./data/detection/"


