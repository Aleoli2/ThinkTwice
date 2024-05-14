# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
import os.path as osp
from abc import ABCMeta, abstractmethod
from os import PathLike
from typing import List

import mmcv
import numpy as np
from torch.utils.data import Dataset


from mmdet3d.datasets.pipelines import Compose

def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


class BaseDataset(Dataset):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """


    def __init__(self):
        super(BaseDataset, self).__init__()
        self.data_length = 0

    @abstractmethod
    def load_annotations(self):
        return mmcv.load(self.ann_file)


    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        pass


    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 jsonfile_prefix=None,
                 indices=None,
                 logger=None):
        pass

