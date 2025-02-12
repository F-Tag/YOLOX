#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp

from datetime import datetime

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # tiny settings
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        # self.random_size = (10, 20)
        self.multiscale_range = 5
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0] + datetime.now().strftime('_%Y%m%d%H%M%S')
        # self.enable_mixup = False

        # dataloader settings
        self.data_num_workers = 12

        # fine tuning
        self.act = "relu6"
        # self.num_classes = 14
        self.num_classes = 20
        self.min_lr_ratio = 0.05 / 100
        self.basic_lr_per_img = 0.01 / 64.0 / 100

        self.degrees = 5.0
        self.shear = 1.0
        self.enable_mixup = True

        # for debug
        # self.eval_interval = 1
        # self.max_epoch = 50

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform

        return VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "2d_bbox_from_3d"),
            # image_sets=[('2007', 'train')],
            image_sets=[('2012', 'train')],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import VOCDetection, ValTransform
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "2d_bbox_from_3d"),
            # image_sets=[('2007', 'test')],
            image_sets=[('2012', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )

