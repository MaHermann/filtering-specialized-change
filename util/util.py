import numpy as np
import torch

def iou(prediction, reference):
    '''
        Compute Intersection Over Union for a given prediction and reference.

        Assumes both tensors have only 0 and 1 entries.
    '''
    intersection = torch.mul(prediction, reference)
    union = torch.max(prediction, reference)
    return intersection.sum() / union.sum()

def false_positive(prediction, reference):
    '''Compute amount of false positives for a given prediction and reference.'''
    return torch.nn.ReLU()(prediction - reference).sum()

def false_negative(prediction, reference):
    '''Compute amount of false negatives for a given prediction and reference.'''
    return torch.nn.ReLU()(reference - prediction).sum()

def compare_to_color(image, color):
    '''
        Create a numpy mask that is `True` when a given pixel in `image` is
        equal to `color` and `False` otherwise.
    '''
    color = np.expand_dims(color, axis=(0, 1))
    return np.all(image == color, axis=-1)
