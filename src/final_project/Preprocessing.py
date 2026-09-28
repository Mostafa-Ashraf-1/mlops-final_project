from typing import Callable, Optional

from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

def open_image(path:str)->Image:

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def parse_breed(fname):
    parts = fname.split('_')
    return ' '.join(parts[:-1])

class PetsDataset(Dataset):
    def __init__(self, root, transform:Callable | None = None):
        super().__init__()
        self.root = root
        self.files = [fname for fname in os.listdir(root) if fname.endswith('.jpg')]
        self.classes = sorted(list({parse_breed(fname) for fname in self.files}))
        self.class_to_idx = {
        cls_name: idx for idx, cls_name in enumerate(self.classes)
        }
        self.transform = transform

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        fpath = os.path.join(self.root, fname)
        img = open_image(fpath)

        if self.transform is not None:
            img = self.transform(img)

        label_str = parse_breed(fname)
        class_idx = self.class_to_idx[label_str]
        return img, class_idx