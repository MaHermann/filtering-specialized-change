import os
from glob import glob
from math import ceil
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image

from util import compare_to_color

class SECOND(Dataset):
    '''
        Implementation of the SECOND dataset (https://arxiv.org/pdf/2010.05687.pdf).


        Args:
            - root: the location of the data. Inside the folder specified here,
              there should be 4 subfolders, 'im1', 'im2', 'label1' and 'label2'
            - patch_side: the dimension of the sub-patches that are created from
              every image
            - stride: the stride between the subpatches. If stride < patch_side,
              there will be overlap between the patches, which can be desirable,
              e.g., to prevent edge effects
            - split: one of 'test', 'train' or 'val', deciding which files to use
            - transform: an optional transform to apply to the images (e.g, rotation
              or mirroring). Even when preloading the data, this will be applied
              to an image every time it is accessed, so that random transformations
              are not fixed for each epoch.
            - preload_dataset: if True, the dataset will be stored in the memory
              when calling the constructor, taking some time but eliminating the
              need for disk accesses later.
            - restrict_change_from: accepts a tuple (so for one type, write, e.g.,
              `restrict_change_from=('tree',)`) so that change is only marked if
              the label1 (the 'from' in 'from - to') is one of the given categories.
              If set to `None`, all categories are seen as change in 'from' (default).
            - restrict_change_to: the corresponding 'to' part of 'restrict_change_from'
            - limit_to_first_n: If set to a positive integer n, only the first n
              images of the dataset are returned. For the whole dataset, set this
              to `None` (default). This is mainly useful for debugging purposes, when loading
              the full dataset would take too long.
            - display_progress_bar: When set to 'True' (default), a progress bar
              is displayed while preloading the dataset.
    '''

    splits = ['test', 'train', 'val']
    # This is mainly to prevent overlap (for detail, see the readme)
    indices = {
        'test':  list(range(2648, 2968)) + [1955, 1964, 2186, 2232],
        'train': list(range(   0, 1215)) + list(range(1216, 1927)) +
                 list(range(1928, 1955)) + list(range(1956, 1964)) +
                 list(range(1965, 2186)) + list(range(2187, 2232)) +
                 list(range(2233, 2374)),
        'val':   list(range(2374, 2648)) + [1215, 1927],
    }

    # the color used for the corresponding category in the label images
    categories = {
        'non-change':       [255, 255, 255],
        'low vegetation':   [  0, 128,   0],
        'n.v.g. surface':   [128, 128, 128],
        'tree':             [  0, 255,   0],
        'water':            [  0,   0, 255],
        'building':         [128,   0,   0],
        'playground':       [255,   0,   0],
    }

    image_shape = (512, 512)

    def __init__(
        self,
        root: str = 'data/SECOND/',
        patch_side: int = 512,
        stride: int = 512,
        split: str = 'train',
        transform: Compose = Compose([lambda x: x]),
        preload_dataset: bool = False,
        restrict_change_from: Optional[Tuple[str, ...]] = None,
        restrict_change_to: Optional[Tuple[str, ...]] = None,
        limit_to_first_n: Optional[int] = None,
        display_progress_bar: bool = True,
    ):
        assert split in self.splits

        self.split = split
        self.transform = transform
        self.preload_dataset = preload_dataset
        self.restrict_change_from = restrict_change_from
        self.restrict_change_to = restrict_change_to

        self.coordinates = self.compute_coordinates(patch_side, stride)
        self.n_patches_per_file = len(self.coordinates)

        self.files = self.load_files(root)

        self.files = [self.files[i] for i in self.indices[split]]

        if limit_to_first_n:
            self.files = self.files[:limit_to_first_n]

        # if we use all change, one label is enough as we have a 'no change' category
        if self.restrict_change_from is None and restrict_change_to is None:
            self.files = [(image1, image2, labels1)
                          for (image1, image2, labels1, _) in self.files]

        if self.preload_dataset:
            self.data = []
            for file in tqdm(self.files, disable=(not display_progress_bar)):
                image1 = self.open_file(file[0])
                image2 = self.open_file(file[1])
                label1 = self.open_file(file[2])
                if (self.restrict_change_from is not None
                 or self.restrict_change_to is not None):
                    label2 = self.open_file(file[3])
                    mask = self.compute_change_mask_restricted(
                        label1,
                        label2,
                        self.restrict_change_from,
                        self.restrict_change_to,
                    )
                else:
                    mask = self.compute_change_mask(label1)
                for i, j, k, l in self.coordinates:
                    self.data.append((image1[i:j, k:l], image2[i:j, k:l], mask[i:j, k:l]))

    @staticmethod
    def load_files(root: str):
        '''Loads the files located at `root`.'''

        images1_root = os.path.join(root, 'im1')
        images2_root = os.path.join(root, 'im2')
        labels1_root = os.path.join(root, 'label1')
        labels2_root = os.path.join(root, 'label2')

        images1 = sorted(glob(os.path.join(images1_root, '*')))
        files = []
        for image1_path in images1:
            filename = image1_path.split(os.sep)[-1]
            image2_path = os.path.join(images2_root, filename)
            labels1_path = os.path.join(labels1_root, filename)
            labels2_path = os.path.join(labels2_root, filename)
            files.append((image1_path, image2_path, labels1_path, labels2_path))
        return files

    @staticmethod
    def open_file(file_path):
        '''Open and return the given file.'''
        image = np.array(Image.open(file_path))
        return image

    @staticmethod
    def compute_change_mask_restricted(label1, label2, categories_from, categories_to):
        '''Computes the change mask, considering only changes from `category_from` to `category_to`.'''

        if categories_from is None:
            change_mask_from = SECOND.compute_change_mask(label1)
        else:
            values_from = [SECOND.categories[category] for category in categories_from]
            change_masks_from = [compare_to_color(label1, value) for value in values_from]
            change_mask_from = np.logical_or.reduce(change_masks_from)

        if categories_to is None:
            change_mask_to = SECOND.compute_change_mask(label2)
        else:
            values_to = [SECOND.categories[category] for category in categories_to]
            change_masks_to = [compare_to_color(label2, value) for value in values_to]
            change_mask_to = np.logical_or.reduce(change_masks_to)

        change_mask = np.logical_and(change_mask_from, change_mask_to).astype(int)
        return change_mask

    @staticmethod
    def compute_change_mask(label):
        '''Computes the change mask, based on only one label and its 'no change' category.'''
        inverted_change_mask =  compare_to_color(
            label,
            SECOND.categories['non-change']
        ).astype(int)
        return 1 - inverted_change_mask

    @staticmethod
    def compute_coordinates(patch_side, stride):
        '''
            Computes the coordinates for each image.

            As each image in the data set has the same dimensions, this needs
            to be done only once and is fairly straightforward.
        '''
        n_patches_per_row = ceil((SECOND.image_shape[0] - patch_side + 1) / stride)
        n_patches_per_col = ceil((SECOND.image_shape[1] - patch_side + 1) / stride)

        coordinates = [
            (
                stride * i,
                stride * i + patch_side,
                stride * j,
                stride * j + patch_side,
            )
            for i in range(n_patches_per_row)
            for j in range(n_patches_per_col)
        ]
        return coordinates


    def __len__(self):
        return len(self.files) * self.n_patches_per_file

    def __getitem__(self, idx):
        """
            Returns image and ground truth with index idx as a tuple (x, mask)
            (see the readme for a detailed discussion of indices, here we use the
            dataset indexing per definition)

            Shape:
                x: :math:`(2, 3, H, W)` with :math:`(H, W)` the shape of the image
                mask: :math:`(H, W)`
        """

        if self.preload_dataset:
            image1, image2, mask = self.data[idx]
        else:
            file_idx = idx // self.n_patches_per_file
            i, j, k, l = self.coordinates[idx % self.n_patches_per_file]
            file = self.files[file_idx]
            image1 = self.open_file(file[0])[i:j, k:l]
            image2 = self.open_file(file[1])[i:j, k:l]
            label1 = self.open_file(file[2])
            if (self.restrict_change_from is not None
             or self.restrict_change_to is not None):
                label2 = self.open_file(file[3])
                mask = self.compute_change_mask_restricted(
                    label1,
                    label2,
                    self.restrict_change_from,
                    self.restrict_change_to,
                )[i:j, k:l]
            else:
                mask = self.compute_change_mask(label1)[i:j, k:l]

        image1 = torch.from_numpy(image1.astype('float32')).float().permute(2,0,1)
        image2 = torch.from_numpy(image2.astype('float32')).float().permute(2,0,1)
        mask = torch.from_numpy(mask.astype('long')).long()

        x = torch.stack([image1, image2], dim=0)

        return self.transform([x, mask])
