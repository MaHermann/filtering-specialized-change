import pytorch_lightning as pl
from torchvision.utils import make_grid

class BatchPredictionMonitor(pl.Callback):
    '''
        Computes and logs predictions of the validation batches.

        The exact prediction is specified by the passed visualization_function.
        This might e.g. just be something like
            ``lambda pl_module, batch: pl_module.make_prediction(batch[0]).unsqueeze(1)``
        or a more complicated function. However, as the output will be passed
        to torchvision.utils.make_grid, it has to be either a B X C X H X W Tensor
        or a list of images
    '''
    def __init__(
        self,
        visualization_function,
        log_every_n_batches=1,
        title='prediction',
        pad_value=100,
        padding=2,
    ):
        self.log_every_n_batches = log_every_n_batches
        self.title = title
        self.padding = padding
        self.pad_value = pad_value
        self.visualization_function = visualization_function


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % self.log_every_n_batches == 0:
            logger = trainer.logger
            if logger:
                img = self.visualization_function(pl_module, batch)
                img = make_grid(
                    img,
                    padding=self.padding,
                    pad_value=self.pad_value,
                )
                str_title = f'{self.title}_{batch_idx}'
                logger.experiment.add_image(
                    str_title,
                    img,
                    global_step=trainer.global_step,
                )
