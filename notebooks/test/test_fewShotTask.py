# %%
import logging
import os

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset
from torchgeo.models import FCSiamConc
from torchvision import transforms
from torchvision.utils import save_image

import util
from datamodules import SECONDDataModule
from datasets import SECOND
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
        images under IMAGES_ROOT/{Name of checkpoint}/{Few-Shot Task}/{description}.
        If N_RUNS is larger than 1, the first run will be saved.

    DISPLAY_PROGRESS_BAR:
        Toggels the progress bar of the tests.

    FEW_SHOT_TASK:
        Sets which of the four defined few-shot tasks we test. Note that, in order
        to be consistent with the paper, we start the indexing at 1. The tasks
        themselves (what categories and what support patches to use) are defined
        in the config.yaml.

    N_SUPPORT:
        How many of the support patches that are defined for the task in the
        config.yaml are used for fine-tuning. If the value is larger than the
        maximum we defined, the script will throw an error during execution.

    MODE:
        Can be set to either 'test' or 'val', depending on what set the tests
        should be run on.

    N_RUNS:
        Sets how many individual test runs are done for each trained model. This
        is mainly to account for the fact that fine-tuning is probabilistic, and
        therefore we can get quite different results for two tests of the same model.

    TODO One way to make testing more consistent would be to also set the random
    seed during testing. This would not change the need for multiple tests runs
    to asses the variance, but the results would be more consistent between
    individual runs of this test script.

    METHOD:
        Decides the method we are testing. Can be set to either finetune,
        baselineA (no finetuning) or baselineB (support-set only).

    DROPOUT:
        Whether to use dropout during fine-tuning.

    DROPOUT_RATE:
        The dropout rate to be used during fine-tuning.

    N_FINETUNE_EPOCHS:
        Number of epochs during fine-tuning.

    FINETUNE_LR:
        Learning rate during fine-tuning.

    CHANGE_WEIGHT:
        The weight used in the loss function for changed pixels in contrast to
        unchanged ones.

    CHECKPOINTS:
        List of names of the individual checkpoints to be tested.
        If created by train_SECOND.py, these will be {MODEL_NAME}_{MODEL_NUMBER}.
        In any case, you don't need the .ckpt ending, and they should be contained
        in the checkpoint directory specified in the config.yaml, without subfolders.
        When METHOD = baselineB, these are not used to load models (as baseline B
        only trains with the support set), but it still needs to be set to as many
        dummy-strings (e.g, 'model_0', 'model_1', etc.) as we want to test for
        (divided by N_RUNS). If you want to compare 5 models with 2 runs each to
        an equal number of baseline B runs, then keep N_RUNS at 2 and insert 5
        dummy models here for the baseline B test.
'''

SAVE_IMAGES = True
DISPLAY_PROGRESS_BAR = False

FEW_SHOT_TASK = 1
N_SUPPORT = 5
MODE = 'test'
N_RUNS = 2
METHOD = 'finetune'

DROPOUT = False
DROPOUT_RATE = 0.2

N_FINETUNE_EPOCHS = 75
FINETUNE_LR = 5e-4
CHANGE_WEIGHT = 0.1

CHECKPOINTS = [
    'FCSiamConc_0', 'FCSiamConc_1', 'FCSiamConc_2', 'FCSiamConc_3', 'FCSiamConc_4',
    'FCSiamConc_5', 'FCSiamConc_6', 'FCSiamConc_7', 'FCSiamConc_8', 'FCSiamConc_9',
]

CHECKPOINT_PATH = conf['checkpoint_base_directory']

IMAGES_PATH = conf['images_base_directory']

# %%
# These parameters are arguments of the datamodule and the model, so look there
# for further documentation. When changing patch_side or stride, also change
# the PATCHES_PER_IMAGE accordingly!
training_params = {
    'weight_decay': 1e-4,
    'lr': FINETUNE_LR,
    'max_epochs': N_FINETUNE_EPOCHS,
    'lr_scheduler_type': 'exponential',
    'lr_scheduler_gamma': 1.0,                # disable lr scheduling
}
test_set_params = {
    'batch_size': 32,
    'patch_side': 256,
    'stride': 128,
    'preload_dataset': True,
}
support_set_params = {
    'patch_side': 256,
    'stride': 256,
    'preload_dataset': True,
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

# How many patches are in an image depends on patch_side and stride (e.g. for
# patch size 256 and stride 128, we will get  9 patches out of the image),
# but instead of calculating it, it is easiest to just set it manually here.
PATCHES_PER_IMAGE = 9

description = (
    METHOD + '+' + MODE  +
    ('' if METHOD == 'baselineA'
     else f'+epochs={N_FINETUNE_EPOCHS}+change_weight={CHANGE_WEIGHT}+dropout={DROPOUT}')
)

# %%
assert METHOD in ['baselineA', 'baselineB', 'finetune']
assert FEW_SHOT_TASK in [1, 2, 3, 4]
assert MODE in ['val', 'test']

# %% Create the necessary folders
if SAVE_IMAGES:
    for checkpoint in CHECKPOINTS:
        os.makedirs(os.path.join(
            IMAGES_PATH, checkpoint, f'few_shot_task_{FEW_SHOT_TASK}', description
        ),
        exist_ok=True)

# %% Suppress output
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# %%
# In the paper, we number the tasks from 1 to 4
task = conf['datasets']['SECOND']['few_shot_tasks'][FEW_SHOT_TASK-1]
if N_SUPPORT > len(task['idx_support']):
    raise ValueError(f'N_SUPPORT ({N_SUPPORT}) cannot be larger than the number '
                     f'of support patches specified in the config.yaml '
                     f'({len(task["idx_support"])})!')
support_dataloader = DataLoader(
    Subset(
        SECOND(
            root=conf['datasets']['SECOND']['path'],
            split='val',
            transform=transforms.Compose([augmentation, transform]),
            restrict_change_from=tuple(task['change_types_from'])
                        if task['change_types_from'] is not None else None,
            restrict_change_to=tuple(task['change_types_to'])
                        if task['change_types_to'] is not None else None,
            **support_set_params,
        ),
        task['idx_support'][:N_SUPPORT],
    ),
    batch_size=N_SUPPORT,
    shuffle=True,
    num_workers=conf['machine']['num_workers'],
)
datamodule = SECONDDataModule(
    root=conf['datasets']['SECOND']['path'],
    num_workers=conf['machine']['num_workers'],
    transform=transform,
    restrict_change_from=tuple(task['change_types_from'])
            if task['change_types_from'] is not None else None,
    restrict_change_to=tuple(task['change_types_to'])
            if task['change_types_to'] is not None else None,
    **test_set_params,
)
# %%
if MODE == 'val':
    dataloader = datamodule.val_dataloader()
else:
    dataloader = datamodule.test_dataloader()
gts = [y for _, y in dataloader]
gts = torch.cat(gts)
scores = []

# %%
for checkpoint in CHECKPOINTS:
    for i in range(N_RUNS):
        model = ChangeDetectionModel(
            FCSiamConc(),
            change_weight=CHANGE_WEIGHT,
            backbone_dropout=DROPOUT_RATE if DROPOUT else 0.0,
            **training_params,
        )
        if METHOD != 'baselineB':               # baseline B means no base training
            checkpoint_state = torch.load(
                os.path.join(CHECKPOINT_PATH, checkpoint + '.ckpt'),
                map_location=(lambda storage, _: storage.cuda(*conf['machine']['gpus']))
                    if conf['machine']['gpus'] is not None
                    else torch.device('cpu'),
            )
            model.load_state_dict(checkpoint_state['state_dict'])

        # We need to set the change weight like this, as it is a parameter and
        # will be loaded from the checkpoint, overwriting any arguments passed
        # in the constructor.
        model.weight = torch.tensor(CHANGE_WEIGHT)

        if METHOD == 'baselineA':             # baseline A means no finetuning
            pass
        else:
            trainer = Trainer(
                gpus=conf['machine']['gpus'],
                logger=False,
                enable_progress_bar=DISPLAY_PROGRESS_BAR,
                enable_model_summary=False,
                enable_checkpointing=False,
                max_epochs=training_params['max_epochs'],
            )
            trainer.fit(model, train_dataloaders=support_dataloader)

        model.prediction_mode = 'binary'
        model.eval()

        trainer = Trainer(
            gpus=conf['machine']['gpus'],
            logger=False,
            enable_progress_bar=DISPLAY_PROGRESS_BAR,
            enable_model_summary=False,
            max_epochs=0,
        )
        if MODE == 'val':
            loader = datamodule.val_dataloader()
        else:
            loader = datamodule.test_dataloader()
        predictions = trainer.predict(model, dataloaders=loader)
        predictions = torch.cat(predictions)

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
        if SAVE_IMAGES and i==0:
            if MODE == 'val':
                loader = datamodule.val_dataloader()
            else:
                loader = datamodule.test_dataloader()
            for i in range(gts.shape[0]):
                img_idx = i // PATCHES_PER_IMAGE
                patch_idx = i % PATCHES_PER_IMAGE
                img_idx = loader.dataset.files[img_idx][0][-9:-4]
                save_image(
                    create_colorcoded_prediction(
                        predictions[i,...],
                        gts[i,...],
                    ).float(),
                    os.path.join(
                        IMAGES_PATH, checkpoint, f'few_shot_task_{FEW_SHOT_TASK}',
                        description, f'{img_idx}_{patch_idx}.png',
                    ),
                )

# %% Summary
print(f'Task: {FEW_SHOT_TASK}\t\tLearning Rate: {FINETUNE_LR}\n{description}')
for i, score in enumerate(['iou', 'prec', 'rec', 'f1']):
    values = [full[i] for full in scores]
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    max = np.max(values)
    min = np.min(values)
    print(f'{score}:\t'
        + f'{mean*100:.2f} ({std*100:.2f})\t'
        + f'{min*100:.2f} - {max*100:.2f}\t')
