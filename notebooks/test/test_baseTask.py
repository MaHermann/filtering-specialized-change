# %%
import logging
import os

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from torchgeo.models import FCSiamConc
from torchvision import transforms
from torchvision.utils import save_image

import util
from datamodules import SECONDDataModule
from modules import ChangeDetectionModel
from util import create_colorcoded_prediction, false_negative, false_positive, iou

# %%
def compute_metrics(prediction, ground_truth):
    fp = false_positive(prediction, ground_truth)
    fn = false_negative(prediction, ground_truth)
    tp = ground_truth.sum() - fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*tp / (2*tp + fp + fn)
    return precision, recall, f1

# %%
conf = yaml.safe_load(open('config.yaml','r'))

# %%
'''
    SAVE_IMAGES:
        If set to True, the results of the test run will be saved as colorcoded
        images under IMAGES_ROOT/{Name of checkpoint}/baseTask.

    DISPLAY_PROGRESS_BAR:
        Toggels the progress bar of the tests.

    CHECKPOINTS:
        List of names of the individual checkpoints to be tested.
        If created by train_SECOND.py, these will be {MODEL_NAME}_{MODEL_NUMBER}.
        In any case, you don't need the .ckpt ending, and they should be contained
        in the checkpoint directory specified in the config.yaml, without subfolders.
'''

SAVE_IMAGES = True

DISPLAY_PROGRESS_BAR = False

CHECKPOINTS = [
    'FCSiamConc_0', 'FCSiamConc_1', 'FCSiamConc_2', 'FCSiamConc_3', 'FCSiamConc_4',
    'FCSiamConc_5', 'FCSiamConc_6', 'FCSiamConc_7', 'FCSiamConc_8', 'FCSiamConc_9',
]

CHECKPOINT_PATH = conf['checkpoint_base_directory']

IMAGES_ROOT = conf['images_base_directory']

# %%
# These parameters are arguments of the datamodule, so look there for further
# documentation. When changing patch_side or stride, also change the PATCHES_PER_IMAGE
# accordingly!
test_set_params = {
    'batch_size': 32,
    'patch_side': 256,
    'stride': 128,
    'preload_dataset': True,
}
transform = transforms.Compose([
    util.NormalizeImg(
        SECONDDataModule.means,
        SECONDDataModule.stds,
    ),
])

# How many patches are in an image depends on patch_side and stride (e.g. for
# patch size 256 and stride 128, we will get  9 patches out of the image),
# but instead of calculating it, it is easiest to just set it manually here.
PATCHES_PER_IMAGE = 9

# %% Create the necessary folders
if SAVE_IMAGES:
    for checkpoint in CHECKPOINTS:
        os.makedirs(os.path.join(IMAGES_ROOT, checkpoint, 'baseTask'), exist_ok=True)

# %% Suppress output
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# %%
datamodule = SECONDDataModule(
    root=conf['datasets']['SECOND']['path'],
    num_workers=conf['machine']['num_workers'],
    transform=transform,
    restrict_change_from=None,
    restrict_change_to=None,
    **test_set_params,
)
gts = [y for _, y in datamodule.test_dataloader()]
gts = torch.cat(gts)
scores = []

# %%
for checkpoint in CHECKPOINTS:
    model = ChangeDetectionModel(
        FCSiamConc(),
    )
    checkpoint_state = torch.load(
        os.path.join(CHECKPOINT_PATH, checkpoint + '.ckpt'),
        map_location=(lambda storage, _: storage.cuda(*conf['machine']['gpus']))
                if conf['machine']['gpus'] is not None
                else torch.device('cpu'),
    )
    model.load_state_dict(checkpoint_state['state_dict'])
    model = model.eval()
    model.prediction_mode = 'binary'

    trainer = Trainer(
        gpus=conf['machine']['gpus'],
        logger=False,
        enable_progress_bar=DISPLAY_PROGRESS_BAR,
        enable_model_summary=False,
        max_epochs=0,
    )
    predictions = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    predictions = torch.cat(predictions)

    # Compute metrics
    iou_score = iou(predictions, gts)
    precision, recall, f1 = compute_metrics(predictions, gts)

    scores.append([iou_score, precision, recall, f1])

    print(
        f'{checkpoint}:\n'
      + f'iou:\t{iou_score*100:.4f}\n'
      + f'prec:\t{precision*100:.4f}\n'
      + f'rec:\t{recall*100:.4f}\n'
      + f'f1:\t{f1*100:.4f}'
    )

    # Create and save images
    if SAVE_IMAGES:
        for i in range(gts.shape[0]):
            img_idx = i // PATCHES_PER_IMAGE
            patch_idx = i % PATCHES_PER_IMAGE
            img_idx = datamodule.test_dataloader().dataset.files[img_idx][0][-9:-4]
            save_image(
                create_colorcoded_prediction(
                    predictions[i,...],
                    gts[i,...],
                ).float(),
                os.path.join(
                    IMAGES_ROOT, checkpoint, 'baseTask', f'{img_idx}_{patch_idx}.png'
                ),
            )

# %% Summary
print(f'\tBase Task')
for i, score in enumerate(['iou', 'prec', 'rec', 'f1']):
    values = [full[i] for full in scores]
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    max = np.max(values)
    min = np.min(values)
    print(f'{score}:\t'
        + f'{mean*100:.2f} ({std*100:.2f})\t'
        + f'{min*100:.2f} - {max*100:.2f}\t')
