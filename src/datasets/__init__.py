import logging
import random
from itertools import chain
from pathlib import Path
from pprint import pprint

import albumentations as albu
import numpy
import torch
from torchsampler import ImbalancedDatasetSampler

from src.datasets.kits_dataset import KITSDataset
from src.datasets.lungs import LungsDataset
from src.datasets.tsm_synth_dataset import TSMSyntheticDataset
from src.datasets.tuh_dataset import TUHDataset
from src.datasets.TCGA_dataset import TCGADataset
from src.utils.generic_utils import seed_everything
from src.datasets.tuh_dataset_paired import TUHDataset_pairs
from src.datasets.black_squares_dataset import BlackSquaresDataset
from src.datasets.tuh_DINO_dataset import TUHDataset_DINO


def build_dataset(kind: str, root_dir: Path, split: str, transforms:albu.Compose, **kwargs):
    scan_params = kwargs.pop('scan_params', {})
    if kind == 'clf-explain-lungs':
        return LungsDataset(root_dir, split, transforms=transforms, explain_classifier=True)
    if kind == 'clf-train-lungs':
        return LungsDataset(root_dir, split, transforms=transforms, explain_classifier=False)
    elif kind == 'totalsegmentor':
        return TSMSyntheticDataset(root_dir, split, transforms=transforms, limit_scans=kwargs.get('limit_scans', 999999), **scan_params)
    elif kind == 'kits':
        return KITSDataset(root_dir, split, transforms=transforms, **scan_params, **kwargs)
    elif kind == 'tcga':
        return TCGADataset(root_dir, split, transforms=transforms, **scan_params, **kwargs)
    elif kind == 'tuh':
        return TUHDataset(root_dir, split, transforms=transforms, **scan_params, **kwargs)
    elif kind == 'tuh_pairs':
        return TUHDataset_pairs(root_dir, split, transforms=transforms, **scan_params, **kwargs)
    elif kind == 'squares':
        return BlackSquaresDataset()
    elif kind == 'DINO':
        return TUHDataset_DINO(root_dir, split, transforms=transforms, **scan_params, **kwargs)
    elif kind == 'merged':
        logging.info(f'HERE ARE THE DATASET CONFIGS: {kwargs["datasets"]}')
        return MergedDataset(split, transforms, kwargs['datasets'])
    else:
        raise ValueError(f'Unsupported dataset kind provided: {kind}')


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, split:str, transforms:albu.Compose, dataset_cfgs:list[dict]):
        assert dataset_cfgs, 'No datasets configured'
        # pprint(dataset_cfgs)
        self.split = split
        self.datasets = [build_dataset(split=split, transforms=transforms, **cfg) for cfg in dataset_cfgs]
        self.dataset = torch.utils.data.ConcatDataset(self.datasets)
        self.classes = self.datasets[0].scans[0].classes

    @property
    def scans(self):
        return list(chain.from_iterable(dst.scans for dst in self.datasets))

    def get_sampling_labels(self):
        lbs = list(chain.from_iterable(dataset.get_sampling_labels() for dataset in self.datasets))
        print(f'[Merged dataset] Number of slices with positive sampling label:', sum(lbs))
        return lbs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def seed_worker(worker_id):
    logging.info(f'initial seed {torch.initial_seed()}')
    worker_seed = torch.initial_seed() % 2**32
    seed_everything(worker_seed)


def get_dataloaders(params, data_transforms, sampler_labels=None, seed=42):
    train_data = build_dataset(split='train', transforms=data_transforms['train'], **params)

    sampler_labels_cache = Path(params.root_dir or '', 'sampler_labels.npy')
    use_sampler = params.get('use_sampler', True)
    if use_sampler and sampler_labels is None:
        print('Using imbalanced sampler for training data loader.')
        if sampler_labels_cache.exists() and not params.get('reset_sampler', True):
            sampler_labels = numpy.load(sampler_labels_cache)
        else:
            sampler_labels = train_data.get_sampling_labels() if sampler_labels is None else sampler_labels
            numpy.save(sampler_labels_cache, sampler_labels)
            #print('Cached training sampler labels at:', sampler_labels_cache)

    #print("sampler Labels", sampler_labels)
    # num_samples=int(params.get('batch_size'))*10
    use_only_1_class = params.get('use_only_1_class', False)
    
    if use_only_1_class:
        train_sampler = None
        
        
    else: 
        train_sampler = ImbalancedDatasetSampler(train_data, labels=sampler_labels) if use_sampler else None
        print('Instantiated training dataset for number of samples:', len(train_data))
        
    

    #print('TRAIN SAMPLER IS USED ?', train_sampler)

    test_data = build_dataset(split='test', transforms=data_transforms['val'], **params)
    print('Instantiated validation dataset for number of samples:', len(test_data))

    rng = torch.Generator()
    rng.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        batch_size=params.batch_size,
        pin_memory=True,
        num_workers=params.num_workers, 
        worker_init_fn=seed_worker,
        generator=rng,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=params.batch_size,
        shuffle=params.get('shuffle_test', False),
        pin_memory=True,
        num_workers=params.num_workers,
        worker_init_fn=seed_worker,
        generator=rng,
    )
    return train_loader, test_loader


__all__ = ['build_dataset']
