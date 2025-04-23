from itertools import chain
from pathlib import Path
import os
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.datasets.tsm_scan import CTScan as _CTScan


class CTScan(_CTScan):
    def __getitem__(self, index):
        s = super().__getitem__(index)
        s['image'] = s['image'].transpose(2, 1).flip((1, 2))
        if s['masks'].shape[0] != 0:
            s['masks'] = s['masks'].transpose(2, 1).flip((1, 2))
        return s


class TUHDataset_DINO(torch.utils.data.Dataset):
    def __init__(self, root_dir:str, split:str='train', split_dir:str='splits', limit_scans:int = 99999, **scan_params):
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
        
        global_crops_scale = (0.4, 1.)
        local_crops_scale = (0.05, 0.4)
        
        # Augmentations 
        self.global_transform1 = A.Compose([
            A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 1.0), p=1.0),
            ToTensorV2()
        ])

        self.global_transform2 = A.Compose([
            A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 0.1), p=1.0),
            A.Solarize(threshold=128, p=0.2),
            ToTensorV2()
        ])
        
        self.local_transform = A.Compose([
            A.RandomResizedCrop(96, 96, scale=local_crops_scale, interpolation=cv2.INTER_CUBIC),  # Crop
            A.HorizontalFlip(p=0.5),  # Equivalent to flip in `flip_and_color_jitter`
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),  # Blur
            ToTensorV2()
        ])
        
        


    def get_sampling_labels(self):
        lbs = list(chain.from_iterable(scan.get_sampling_labels() for scan in self.scans))
        print(f'[TUH dataset] Number of slices with positive sampling label:', sum(lbs))
        return lbs
    
    def get_relative_indices(self):
        return list(chain.from_iterable(scan.get_relative_indices(levels = 20) for scan in self.scans))

    def __len__(self):
        return len(self.scans_with_label_1)

    def __getitem__(self, index):
        sample_index = self.scans_with_label_1[index]
        sample = self.scans_dataset[sample_index]
        
        sample_2_index =  np.random.choice(self.scans_with_label_1)
        sample_2 = self.scans_dataset[sample_2_index]
        

        #normalise 
        sample['image'] = (sample['image'] + 1)/2
        sample_2['image'] = (sample_2['image'] + 1)/2
        

        
        if sample['image'].shape[0] == 1:  # Ensure it's a single-channel image
            sample['image'] = sample['image'].repeat(3, 1, 1).numpy()  # Expand to 3 channels
        if sample_2['image'].shape[0] == 1:
            sample_2['image'] = sample_2['image'].repeat(3, 1, 1).numpy()
        
        
        
        crops = []
        
        crops.append(self.global_transform1(image=sample['image']))
        crops.append(self.global_transform2(image=sample['image']))
        crops.append(self.global_transform1(image=sample_2['image']))
        crops.append(self.global_transform2(image=sample_2['image']))
        
        for _ in range(4):
            crops.append(self.local_transform(image=sample_2['image']))
    
        
        
        return crops
