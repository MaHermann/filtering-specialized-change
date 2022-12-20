from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import SECOND


class SECONDDataModule(LightningDataModule):
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
            - augmentation: an optional transform to apply to the images (e.g, rotation
              or mirroring). Even when preloading the data, this will be applied
              to an image every time it is accessed, so that random transformations
              are not fixed for each epoch. This will only be applied to the
              train set.
            - transform: the same as `augmentation`, but applied to all splits,
              so also the test and validation sets. This way, we can have general
              transformations, such as normalization, and augmentations, such as
              mirroring and rotations, that will only be used during training.
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
            - num_workers: how many subprocesses to use for data loading.
              `0` means that the data will be loaded in the main process. (default: `0`)
            - batch_size: how many samples per batch to load (default: `1`).
            - display_progress_bar: When set to 'True' (default), a progress bar
              is displayed while preloading the dataset.
    '''


    # statistics of the dataset (train split only)
    means = [109.3916, 111.6105, 114.5281]
    stds = [43.6571, 40.6855, 40.6918]
    # note that this applies to the standard dataset, i.e. ALL change, not just subtypes.
    percent_change = 0.2019

    def __init__(
        self,
        root: str = 'data/SECOND/',
        patch_side: int = 512,
        stride: int = 512,
        augmentation: Optional[transforms.Compose] = None,
        transform: transforms.Compose = transforms.Compose([lambda x: x]),
        preload_dataset: bool = False,
        restrict_change_from: Optional[Tuple[str, ...]] = None,
        restrict_change_to: Optional[Tuple[str, ...]] = None,
        limit_to_first_n: Optional[int] = None,
        num_workers: int = 0,
        batch_size: int = 1,
        display_progress_bar: bool = True,
    ):
        super().__init__()
        self.root = root
        self.patch_side = patch_side
        self.stride = stride
        self.transform = transform
        self.augmentation = augmentation
        self.preload_dataset = preload_dataset
        self.restrict_change_from = restrict_change_from
        self.restrict_change_to = restrict_change_to
        self.limit_to_first_n = limit_to_first_n
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.display_progress_bar = display_progress_bar
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def test_dataloader(self):
        if self.test_dataset is None:
            self.test_dataset = SECOND(
                root=self.root,
                patch_side=self.patch_side,
                stride=self.stride,
                split='test',
                transform=self.transform,
                preload_dataset=self.preload_dataset,
                restrict_change_from=self.restrict_change_from,
                restrict_change_to=self.restrict_change_to,
                limit_to_first_n=self.limit_to_first_n,
                display_progress_bar=self.display_progress_bar,
            )
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            self.train_dataset = SECOND(
                root=self.root,
                patch_side=self.patch_side,
                stride=self.stride,
                split='train',
                transform=(transforms.Compose([self.augmentation, self.transform])
                            if self.augmentation else self.transform),
                preload_dataset=self.preload_dataset,
                restrict_change_from=self.restrict_change_from,
                restrict_change_to=self.restrict_change_to,
                limit_to_first_n=self.limit_to_first_n,
                display_progress_bar=self.display_progress_bar,
            )
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.val_dataset = SECOND(
                root=self.root,
                patch_side=self.patch_side,
                stride=self.stride,
                split='val',
                transform=self.transform,
                preload_dataset=self.preload_dataset,
                restrict_change_from=self.restrict_change_from,
                restrict_change_to=self.restrict_change_to,
                limit_to_first_n=self.limit_to_first_n,
                display_progress_bar=self.display_progress_bar,
            )
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )
