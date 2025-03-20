import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class BlackScan(Dataset):
    """
    Helper class to mimic the structure of CTScan class in the original codebase.
    """
    def __init__(self, noise_scale, image_size=(256, 256), channel_count=1, num_items=16, mask_channels=3, seed=None):
        self.noise_scale = noise_scale  # Use color_value as scaling factor for the noise
        self.image_size = image_size
        self.channel_count = channel_count
        self.num_items = num_items
        self.mask_channels = mask_channels
        self.classes = ['background', 'object']  # Default classes for compatibility
        self.seed = seed  # For reproducible noise if needed
    
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):
        # Set seed for reproducibility if provided
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)
            
        # Create a random noise image
        # Using randn generates values from a normal distribution (mean=0, std=1)
        # We scale it by noise_scale to control the intensity
        image = self.noise_scale * torch.randn((self.channel_count, self.image_size[0], self.image_size[1]))
        
        # Clip values to stay within -1 to 1 range
        image = torch.clamp(image, -1.0, 1.0)
        
        # Create masks with multiple channels
        masks = torch.zeros((self.mask_channels, self.image_size[0], self.image_size[1]))
        
        # Return a dictionary similar to the CTScan class structure
        return {
            'image': image,
            'masks': masks,
            'label': 0,
            'color_value': self.noise_scale  # Keep this for compatibility
        }
    
    def get_sampling_labels(self):
        return [0] * self.num_items


class BlackSquaresDataset(Dataset):
    def __init__(self, min_scale=0.1, max_scale=1.0, num_scales=10, image_size=(256, 256), 
                 channel_count=1, duplicates=16, mask_channels=3, seed=None):
        """
        Dataset that generates random noise images for model debugging
        
        Args:
            min_scale: Minimum scaling factor for noise (-1 by default)
            max_scale: Maximum scaling factor for noise (1 by default)
            num_scales: Number of different scaling factors between min and max
            image_size: Size of the generated images (height, width)
            channel_count: Number of channels in the generated images
            duplicates: Number of times each noise scale is duplicated (16 by default)
            mask_channels: Number of channels in the mask tensor (3 by default)
            seed: Random seed for reproducibility (None by default)
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_scales = num_scales
        self.image_size = image_size
        self.channel_count = channel_count
        self.duplicates = duplicates
        self.mask_channels = mask_channels
        self.seed = seed
        
        # Generate scale values with even increments
        self.scale_values = torch.linspace(min_scale, max_scale, num_scales)
        
        # Total dataset size
        self.length = num_scales * duplicates
        
        # For compatibility with other code
        self.classes = ['background', 'object']
        
        # Create scans for compatibility with the MergedDataset class
        self.scans = []
        for scale_idx in range(num_scales):
            noise_scale = self.scale_values[scale_idx]
            scan_seed = None if seed is None else seed + scale_idx
            self.scans.append(BlackScan(
                noise_scale, 
                image_size, 
                channel_count, 
                duplicates,
                mask_channels,
                scan_seed
            ))
        
        # Create a ConcatDataset from the scans for __getitem__ compatibility
        self.scans_dataset = ConcatDataset(self.scans)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.scans_dataset[idx]
    
    def get_sampling_labels(self):
        # Concatenate labels from all scans
        return [0] * self.length
