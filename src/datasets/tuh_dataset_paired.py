from itertools import chain
from pathlib import Path
import os
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import ConcatDataset

from src.datasets.tsm_scan import CTScan as _CTScan


class CTScan(_CTScan):
    def __getitem__(self, index):
        s = super().__getitem__(index)
        s['image'] = s['image'].transpose(2, 1).flip((1, 2))
        if s['masks'].shape[0] != 0:
            s['masks'] = s['masks'].transpose(2, 1).flip((1, 2))
        return s


class TUHDataset_pairs(torch.utils.data.Dataset):
    def __init__(self, root_dir:str, split:str, split_dir:str='splits', limit_scans:int = 99999, **scan_params):
        self.root_dir = Path(root_dir)
        self.split = split
        
        #self.ann_path = self.root_dir / split_dir / f'{split}_scans.csv'
        self.ann_path = os.path.join(split_dir, f'{split}_scans.csv')
        #assert self.ann_path.exists()
        with open(self.ann_path, 'r') as fid:
            scan_names = set(fid.read().splitlines())
        assert scan_names, f'No scans found in split: {self.ann_path}'
        
        
        # THIS IS ALL BECAUSE OF THE WAY THE DATA IS ORGANIZED
        ## TRAIN
        # CASES
        scans_dir = self.root_dir / 'tuh_train' / 'cases' / 'images' / 'train'
        labels_dir = self.root_dir / 'tuh_train' / 'cases' / 'labels' / 'train'
        self.scans = []
        for i, sp in enumerate(scans_dir.rglob('*.nii.gz')):
            if i > limit_scans:
                break
            sname = sp.name.replace('_0000', '')
            if sname not in scan_names:
                continue
            # self.scans.append(CTScan(sp, labels_dir / sp.parent.name / sname, **scan_params))
            self.scans.append(CTScan(sp, labels_dir / sname, **scan_params))


        # CONTROLS
        scans_dir = self.root_dir / 'tuh_train' / 'controls' / 'images' / 'train'
        labels_dir = self.root_dir / 'tuh_train' / 'controls' / 'labels' / 'train'
        for i, sp in enumerate(scans_dir.rglob('*.nii.gz')):
            if i > limit_scans:
                break
            sname = sp.name.replace('_0000', '')
            if sname not in scan_names:
                continue
            # self.scans.append(CTScan(sp, labels_dir / sp.parent.name / sname, **scan_params))
            self.scans.append(CTScan(sp, labels_dir / sname, **scan_params))
        
        ### TEST
        # CASES
        scans_dir = self.root_dir / 'tuh_test' / 'cases' / 'images' / 'test'
        labels_dir = self.root_dir / 'tuh_test' / 'cases' / 'labels' / 'test'

        for i, sp in enumerate(scans_dir.rglob('*.nii.gz')):
            if i > limit_scans:
                break
            sname = sp.name.replace('_0000', '')
            if sname not in scan_names:
                continue
            # self.scans.append(CTScan(sp, labels_dir / sp.parent.name / sname, **scan_params))
            self.scans.append(CTScan(sp, labels_dir / sname, **scan_params))

        #CONTROLS 
        scans_dir = self.root_dir / 'tuh_test' / 'controls' / 'images' / 'test'
        labels_dir = self.root_dir / 'tuh_test' / 'controls' / 'labels' / 'test'

        for i, sp in enumerate(scans_dir.rglob('*.nii.gz')):
            if i > limit_scans:
                break
            sname = sp.name.replace('_0000', '')
            if sname not in scan_names:
                continue
            # self.scans.append(CTScan(sp, labels_dir / sp.parent.name / sname, **scan_params))
            self.scans.append(CTScan(sp, labels_dir / sname, **scan_params))

        # Remove elements from self.scans that have length 0

        self.scans_dataset = ConcatDataset(self.scans)
        self.classes = self.scans[0].classes
        sampling_labels = self.get_sampling_labels()
        self.scans_with_label_1 = [i for i, label in enumerate(sampling_labels) if label == 1]
        self.scans_with_label_0 = [i for i, label in enumerate(sampling_labels) if label == 0]
        
        self.relative_slice_height = self.get_relative_indices()
        
        scans_with_label_0_heights = defaultdict(list)
        
        for i, index in enumerate(self.scans_with_label_0):
            height = self.relative_slice_height[index]
            scans_with_label_0_heights[height].append(index)
            
        self.scans_with_label_0_heights = scans_with_label_0_heights


    def get_sampling_labels(self):
        lbs = list(chain.from_iterable(scan.get_sampling_labels() for scan in self.scans))
        print(f'[TUH dataset] Number of slices with positive sampling label:', sum(lbs))
        return lbs
    
    def get_relative_indices(self):
        return list(chain.from_iterable(scan.get_relative_indices() for scan in self.scans))

    def __len__(self):
        return len(self.scans_with_label_1)

    def __getitem__(self, index):
        sample_index = self.scans_with_label_1[index]
        sample = self.scans_dataset[sample_index]
        
        sample_height = self.relative_slice_height[sample_index]

        
        
        healthy_index = np.random.choice(self.scans_with_label_0_heights[sample_height])
        
        healthy_sample = self.scans_dataset[healthy_index]
        sample['healthy_example'] = healthy_sample['image']
        return sample
