# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import List
import os.path as osp
import mmengine
import pickle
import numpy as np

@DATASETS.register_module()
class UDA_Dataset_ISPRS(BaseSegDataset):
    """UDA dataset for ISPRS.

    In segmentation map annotation for Potsdam/Vaihingen dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('impervious_surface', 'building', 'low_vegetation', 'tree',
                 'car', 'clutter'),
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 B_img_suffix='.png',
                 B_img_path=None,
                 B_img_file='',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        self.B_img_path = B_img_path
        self.B_img_suffix = B_img_suffix
        self.B_img_file = osp.join(self.data_root, B_img_file)
        self.B_data_list: List[dict] = []
        if self.B_img_path is not None:
            self.B_data_list = self.load_B_data_list()
        else:
            self.B_data_list = None
    
    ## added by LYU: 2024/04/15
    ## target image info
    def load_B_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        if not osp.isdir(self.B_img_file) and self.B_img_file:
            assert osp.isfile(self.B_img_file), \
                f'Failed to load `B_img_file` {self.B_img_file}'
            lines = mmengine.list_from_file(
                self.B_img_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    B_img_path=osp.join(self.data_root, self.B_img_path, img_name + self.B_img_suffix))
                data_list.append(data_info)
        else:
            _suffix_len = len(self.B_img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=self.B_img_path,
                    list_dir=False,
                    suffix=self.B_img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(B_img_path=osp.join(self.data_root, self.B_img_path, img))
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['B_img_path'])
        return data_list
    
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        
        ## added by LYU: 2024/04/16
        if self.B_data_list is not None:
            assert len(self.B_data_list) > 0
            idx_b = np.random.randint(0, len(self.B_data_list))
            data_info['B_img_path'] = self.B_data_list[idx_b]['B_img_path']

        return data_info