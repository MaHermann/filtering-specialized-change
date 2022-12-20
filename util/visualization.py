import torch
from PIL import Image


def create_colorcoded_prediction(prediction, ground_truth):
    '''Create a standard visualization for quick assessment of performance.'''

    true_positive = torch.logical_and(ground_truth, prediction)
    false_positive = torch.nn.ReLU()(prediction - ground_truth)
    false_negative = torch.nn.ReLU()(ground_truth - prediction)

    # plot false postivies in red, false negatives in blue
    prediction_image = torch.stack((true_positive + false_positive,
                                true_positive,
                                true_positive + false_negative))
    return prediction_image


def image_grid(imgs, rows, cols):
    '''Make a grid with a given number of rows and columns out of the given list of images.'''

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def binary_prediction(pl_module, batch):
    '''Make a binary prediction; to be used as part of a `BatchPredictionMonitor`.'''
    x, y = batch
    predictions = pl_module.make_prediction(x, 'binary')
    images = [
        create_colorcoded_prediction(predictions[i,...], y[i,...])
        for i in range(x.shape[0])      # batch size
    ]
    return images

def change_logit_prediction(pl_module, batch):
    '''Make a logit prediction of the change; to be used as part of a `BatchPredictionMonitor`.'''
    x, _ = batch
    predictions = pl_module.make_prediction(x, 'logits_change')
    predictions = predictions.unsqueeze(1)
    images = [
        predictions[i,...] for i in range(x.shape[0])      # batch size
    ]
    return images
