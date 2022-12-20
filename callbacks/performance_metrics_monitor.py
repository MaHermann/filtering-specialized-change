import numpy as np
import pytorch_lightning as pl
import torch

from util import false_negative, false_positive, iou

class PerformanceMetricsMonitor(pl.Callback):
    '''Compute and log metrics about the predicted clusters in `FewShotModel`.'''

    def __init__(
        self,
        log_every_n_batches=1,
        main_tag='metrics',
    ):
        self.log_every_n_batches = log_every_n_batches
        self.main_tag = main_tag
        self.iou_train = []
        self.tp_train = 0
        self.fp_train = 0
        self.fn_train = 0
        self.iou_val = []
        self.tp_val = 0
        self.fp_val = 0
        self.fn_val = 0
        self.iou_test = []
        self.tp_test = 0
        self.fp_test = 0
        self.fn_test = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_batches == 0:
            pl_module.eval()
            x, y = batch
            predictions = pl_module.make_prediction(x, 'binary')
            if y.sum() > 0 or predictions.sum() > 0:
                self.iou_train.append(iou(predictions, y).cpu())
            # computing the f1 score for every batch and then averaging
            # is not the same as computing it over all of the batches collectively
            fn = false_negative(predictions, y)
            fp = false_positive(predictions, y)
            tp = y.sum() - fn
            self.fn_train += fn
            self.fp_train += fp
            self.tp_train += tp
            pl_module.train()

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log(
            f'{self.main_tag}/iou/train',
            np.mean(self.iou_train),
        )
        pl_module.log(
            f'{self.main_tag}/f1/train',
            2*self.tp_train / (2*self.tp_train + self.fp_train + self.fn_train),
        )
        pl_module.log(
            f'{self.main_tag}/precision/train',
            self.tp_train / (self.tp_train + self.fp_train),
        )
        pl_module.log(
            f'{self.main_tag}/recall/train',
            self.tp_train / (self.tp_train + self.fn_train),
        )
        self.iou_train = []
        self.tp_train = 0
        self.fp_train = 0
        self.fn_train = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        x, y = batch
        predictions = pl_module.make_prediction(x, 'binary')
        if y.sum() > 0 or predictions.sum() > 0:
            self.iou_val.append(iou(predictions, y).cpu())
        else:
            self.iou_val.append(torch.FloatTensor([1]).cpu())
        # computing the f1 score for every batch and then averaging
        # is not the same as computing it over all of the batches collectively
        fn = false_negative(predictions, y)
        fp = false_positive(predictions, y)
        tp = y.sum() - fn
        self.fn_val += fn
        self.fp_val += fp
        self.tp_val += tp

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log(
            f'{self.main_tag}/iou/val',
            np.mean(self.iou_val),
        )
        pl_module.log(
            f'{self.main_tag}/f1/val',
            2*self.tp_val / (2*self.tp_val + self.fp_val + self.fn_val),
        )
        pl_module.log(
            f'{self.main_tag}/precision/val',
            self.tp_val / (self.tp_val + self.fp_val),
        )
        pl_module.log(
            f'{self.main_tag}/recall/val',
            self.tp_val / (self.tp_val + self.fn_val),
        )
        self.iou_val = []
        self.tp_val = 0
        self.fp_val = 0
        self.fn_val = 0
