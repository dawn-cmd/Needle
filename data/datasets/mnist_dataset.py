from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as img_file:
            magic_number, num_examples, row, column = struct.unpack('>4I', img_file.read(16))
            assert(magic_number == 2051)
            input_dim = row * column
            X = np.array(struct.unpack(str(input_dim * num_examples) + 'B', img_file.read()), dtype=np.float32).reshape(num_examples, input_dim)
            X -= np.min(X)
            X /= np.max(X)
        with gzip.open(label_filename, 'rb') as label_file:
            magic_number, num_items = struct.unpack('>2I', label_file.read(8))
            assert(magic_number == 2049)
            y = np.array(struct.unpack(str(num_items) + 'B', label_file.read()), dtype=np.uint8)
        self.images = X
        self.img_row = row
        self.img_column = column
        self.labels = y
        self.transforms = transforms
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # BEGIN YOUR SOLUTION
        imgs = self.images[index]
        labels = self.labels[index]
        if len(imgs.shape) > 1:
            imgs = np.array([self.apply_transforms(img.reshape(self.img_row, self.img_column, 1)).reshape(imgs[0].shape) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape(self.img_row, self.img_column, 1)).reshape(imgs.shape)
        return (imgs, labels)
        # END YOUR SOLUTION

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return self.images.shape[0]
        # END YOUR SOLUTION
