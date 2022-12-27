import os
import glob
import pathlib
from random import random
import numpy as np
import torch
import cv2

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class WM811k(Dataset):
    label2idx = {
        'center'      : 0,
        'donut'       : 1,
        'edge-loc'    : 2,
        'edge-ring'   : 3,
        'loc'         : 4,
        'near-full'   : 5,
        'none'        : 6,
        'random'      : 7,
        'scratch'     : 8,
        '-'           : 9,
    }
    idx2label = [k for k in label2idx.keys()]
    num_classes = len(idx2label) - 1  # exclude unlabeled (-)
    
    def __init__(self, root, transform=None, transform2=None, proportion=1.0, transformtwice: bool = False, **kwargs):
        super(WM811k, self).__init__()
        
        self.root = root
        self.transform = transform
        self.proportion = proportion
        self.transformtwice = transformtwice
        self.transform2 = transform2
        
        images  = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images
        labels  = [pathlib.PurePath(image).parent.name for image in images]
        targets = [self.label2idx[l] for l in labels]
        samples = list(zip(images, targets))
        
        if self.proportion < 1.0:
            # Randomly sample a proportion of the data
            self.samples, _ = train_test_split(
                samples,
                train_size=self.proportion,
                stratify=[s[1] for s in samples],
                shuffle=True,
                random_state=1993 + kwargs.get('seed', 0),
            )
            self.paths, self.targets = zip(*self.samples)
        else:
            self.samples = samples
            self.paths, self.targets = zip(*self.samples)
            
    def __getitem__(self, idx):
        
        path, y = self.samples[idx]
        x = self.load_image_cv2(path)
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y, idx
    
    def __len__(self):
        return len(self.samples)
    
    
    @staticmethod
    def load_image_cv2(filepath: str):
        """Load image with cv2. Use with 'albumentations'."""
        out = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)   # 2D; (H, W)
        return np.expand_dims(out, axis=2)                 # 3D; (H, W, 1)
    
    @staticmethod
    def load_image_pil(filepath: str):
        """Load image with PIL. Use with `torchvision`."""
        return Image.open(filepath)
    

    
class WM811k_valid(WM811k):
    def __init__(self, root, transform=None):
        super(WM811k_valid, self).__init__(root, transform, proportion=1.0)
        self.paths, self.targets = zip(*self.samples)
        
    def __getitem__(self, idx):
        # path, y = self.samples[idx]
        path = self.paths[idx]
        y = self.targets[idx]
        img = self.load_image_cv2(path)
        
        if self.transform is not None:
            x = self.transform(img)
        # return dict(x=x, targets=y, idx=idx)
        return x, y, idx
    
    
    
class WM811k_unlabel(WM811k):
    def __init__(self, root, transform=None, transform2=None, transformtwice: bool = True):
        super(WM811k_unlabel, self).__init__(root, transform, transform2, proportion=1.0, transformtwice=transformtwice)
        self.paths, self.targets = zip(*self.samples)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.targets[idx]
        img = self.load_image_cv2(path)
        
        if self.transformtwice:
            x1 = self.transform(img)
            x2 = self.transform2(img)
            
            
        return (x1, x2), y, idx