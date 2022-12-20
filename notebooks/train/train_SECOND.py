# %%
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchgeo.models import FCSiamConc
from torchvision import transforms

import util
from callbacks import BatchPredictionMonitor, PerformanceMetricsMonitor
from datamodules import SECONDDataModule
from modules import ChangeDetectionModel


# %%
conf = yaml.safe_load(open('config.yaml','r'))

device = 'cuda' if conf['machine']['gpus'] is not None else 'cpu'

# %%
# This sets the name you want to use to identify your trained model. It can be
# used, e.g., to separate different experiment series
MODEL_NAME = 'FCSiamConc'
# The number distinguishes individual models with the same name. Also, the
# random seed for a training run is computed by adding the random seed in the
# config.yaml to the model number, so that models with different number are
# trained with different seeds, and models with the same number are trained with
# the same seed
MODEL_NUMBER = 0

# This gives an option to disable the progress bars during training, e.g., if
# there are issues when saving the ouptut of a run to a text file.
DISPLAY_PROGRESS_BARS = True

TENSORBOARD_PREFIX = f'{MODEL_NAME}/{MODEL_NUMBER}'
MODEL_FILENAME = f'{MODEL_NAME}_{MODEL_NUMBER}'

CHECKPOINT_DIRECTORY = conf['checkpoint_base_directory']

LOG_DIRECTORY = conf['log_base_directory']

# This affects for how many validation batches we log the prediction and logit
# images. Reducing this (i.e., logging more batches) will lead to a larger log file
LOG_EVERY_N_BATCHES = 16

# %%
pl.seed_everything(conf['random_seed'] + MODEL_NUMBER)

# %%
# Here we set the main parameters for the training. Most of them are arguments
# of either the datamodule or the model, so look there for further documentation.
# model_checkpoint_metric and model_checkpoint_mode are arguments of the
# `ModelCheckpoint` callback in PytorchLightning
data_loader_params = {
    'batch_size': 32,
    'patch_side': 256,
    'stride': 128,
    'preload_dataset': True,
    'limit_to_first_n': None,
    'restrict_change_from': None,
    'restrict_change_to': None,
}
model_params = {
    'change_weight': 0.5 *((1 / SECONDDataModule.percent_change) - 1),
    'backbone_dropout': 0.0,
}
training_params = {
    'weight_decay': 1e-4,
    'lr': 5e-4,
    'max_epochs': 100,
    'model_checkpoint_metric': 'loss/overall_loss/val',
    'model_checkpoint_mode': 'min',
    'lr_scheduler_type': 'exponential',
    'lr_scheduler_gamma': 0.95,
}
augmentation = transforms.Compose([
    util.RandomRot(),
    util.RandomFlip(),
])
transform = transforms.Compose([
    util.NormalizeImg(
        SECONDDataModule.means,
        SECONDDataModule.stds,
    ),
])
# %%
datamodule = SECONDDataModule(
    root=conf['datasets']['SECOND']['path'],
    num_workers=conf['machine']['num_workers'],
    augmentation=augmentation,
    transform=transform,
    **data_loader_params,
    display_progress_bar=DISPLAY_PROGRESS_BARS,
)
# %%
model = ChangeDetectionModel(
    FCSiamConc(),
    **model_params,
    **training_params,
    display_epoch_number=(not DISPLAY_PROGRESS_BARS),
)
# %%
logger = TensorBoardLogger(LOG_DIRECTORY, name=TENSORBOARD_PREFIX)
model_checkpoint = ModelCheckpoint(
    monitor=training_params['model_checkpoint_metric'],
    mode=training_params['model_checkpoint_mode'],
    dirpath=CHECKPOINT_DIRECTORY,
    filename=MODEL_FILENAME,
)
binary_prediction_monitor = BatchPredictionMonitor(
    util.binary_prediction,
    title='prediction',
    log_every_n_batches=LOG_EVERY_N_BATCHES,
)
change_logit_monitor = BatchPredictionMonitor(
    util.change_logit_prediction,
    title='change_logit',
    log_every_n_batches=LOG_EVERY_N_BATCHES,
)
# %%
trainer = pl.Trainer(
    callbacks=[
        model_checkpoint,
        binary_prediction_monitor,
        PerformanceMetricsMonitor(),
        change_logit_monitor,
    ],
    max_epochs=training_params['max_epochs'],
    logger=logger,
    gpus=conf['machine']['gpus'],
    enable_progress_bar=DISPLAY_PROGRESS_BARS,
)
# %%
logger.log_hyperparams({
    **model_params,
    **data_loader_params,
    **training_params,
})
# %%
trainer.fit(model, datamodule=datamodule)

# %%
print(f'Best Model Score: {model_checkpoint.best_model_score}')
