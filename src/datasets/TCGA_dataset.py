from itertools import chain
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from src.datasets.tsm_scan import CTScan as _CTScan


class CTScan(_CTScan):
    slicing_dims = {
        'sagittal': 0,  # side view
        'coronal': 1,  # front view
        'axial': 2,  # top down view
    }
    
    def __getitem__(self, index):
        s = super().__getitem__(index)
        s['image'] = s['image'].transpose(2,1).flip((1, 2))
        if s['masks'].shape[0] != 0:
            s['masks'] = s['masks'].transpose(2,1).flip((1, 2))
        return s
        

class TCGADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir:str, split:str, split_dir:str='splits', limit_scans:int = 99999, **scan_params):
        self.root_dir = Path(root_dir)
        self.split = split
        
        self.ann_path = self.root_dir / split_dir / f'{split}_scans.csv'
        assert self.ann_path.exists()
        with open(self.ann_path, 'r') as fid:
            scan_names = fid.read().splitlines()
        assert scan_names, f'No scans found in split: {self.ann_path}'
        
        self.scans = []
        for i, sn in enumerate(scan_names):
            if i > limit_scans:
                break
            self.scans.append(CTScan(self.root_dir / 'imagesTr'/ split / sn.replace('.nii.gz','_0000.nii.gz'), self.root_dir / 'labelsTr'/ split / sn, **scan_params))

        self.scans_dataset = ConcatDataset(self.scans)
        self.classes = self.scans[0].classes
        
    def get_sampling_labels(self):
        lbs = list(chain.from_iterable(scan.get_sampling_labels() for scan in self.scans))
        print(f'[TGCA dataset] Number of slices with positive sampling label:', sum(lbs))
        return lbs

    def __len__(self):
        return len(self.scans_dataset)

    def __getitem__(self, index):
        return self.scans_dataset[index]
