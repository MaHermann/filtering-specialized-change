# Introduction

This is the code for the paper **Filtering Specialized Change in a Few-Shot Setting**. The aim of the experiments is to train a change detection model on all change in a semantic change detection dataset, and then adapt it via fine-tuning on a few samples to a specialized subcategory of change.

![An image showing a schematic illustration of the few-shot filtering idea. We can quickly adapt a base model that is trained on all change to a specialized system via only a few samples. The image shows roadwork and deforestation as two such specialized tasks.](img/Illustrations/few_shot_filtering_schematic.png "Schematic illustration of the few-shot filtering idea")

# Usage

## Dataset
You will need a copy of the [SECOND dataset](http://www.captain-whu.com/project/SCD/). Please cite the corresponding paper when using the dataset for your own work.

The default location for this is a folder named `SECOND` inside `data` in this directory, so with a folder structure looking like this:

    - filtering-specialized-change
        - callbacks
        - data
            -SECOND
                - im1
                - im2
                - label1
                - label2
        - ...

You can also specify a custom path if you want to save this dataset somewhere else. For details see the *The config.yaml* Section down below.

## General Usage

To reproduce the experiments in the paper, use the scripts in the `notebooks` folder. Hyperparameters etc. can be set directly in the files, local information such as gpus and folders can be set in the `config.yaml` and this is also the place where the concrete few-shot tasks are defined. For further description see *The config.yaml* down below.

The most straighforward way to run these is to run, directly in the main project folder,

    python -m notebooks.train.train_SECOND

for training,

    python -m notebooks.test.test_baseTask

for the full model (the line labeled 'Backbone' in Table 1 in the paper) and

    python -m notebooks.test.test_fewShotTask

for the few-shot experiments.

Even though these files are in a folder labeled `notebooks`, these are normal python programs, not Jupyter Notebooks. However, they can be used in a similar way when using VSCode (see below).

## VSCode

The files in the `notebooks` folder are written to be used in a notebook style fashion together with the VSCode Python extension, using the `# %%` syntax that marks new cells.

For this to work properly, you need to set `jupyter.notebookFileRoot` to `${workspaceFolder}` in the VSCode settings so that the imports of the modules work.

## Checkpoints and Logs

The training will write a checkpoint (i.e., a saved model) in the directory specified in the config.yaml after each epoch, choosing the previous epoch with the lowest validation loss.

Additionally, a tensorboard logfile will be written to the logging directory that has been specified in the conf.yaml. To inspect it, use tensorboard, and see their documentation for further info.

## The config.yaml

Most local information (gpu-usage, location of the dataset, etc.) can be configured in the `config.yaml`.

### Workers and GPU usage

`num_workers` is directly passed to the `DataLoader`s that are used for training and testing and therefore works as usual in Pytorch, i.e., `0` means that we use the main process.

For `gpus`, the input needs to be a list (e.g. `[1, 5]` or `[3]`), with the indices corresponding to the gpus as seen by PytorchLightning (it is passed directly to the `gpus` argument of the `Trainer`.) If you don't want to use a gpu, but work on the cpu instead, either leave this blank (`gpus: `) or write `gpus: null`.

### Paths

If you have problems with the paths specified in the config.yaml, try using relative paths (i.e., use `../data/SECOND/` rather than `/home/data/SECOND`) instead.

### The Few-Shot Tasks

The few-shot tasks that are used in testing are defined in the config as well. The list there corresponds to the individual tasks, and we start with the index `1` (instead of `0`) to be compatible with the paper.

![An image showing how individual few-shot tasks are defined. Using the labels of the pre- and postchange images, we can select which change is relevant. There are three cases given: If we use any change type both for pre- and postchange, we get all change. If we specify tree as the prechange, and any change as the postchange image, we get deforestation, and for any change in the prechange, and buildings in the postchange, we get buidling construction.](img/Illustrations/few_shot_tasks.png "Illustration of how to define few-shot tasks")

Few-Shot tasks are defined by their `from` and `to` change types and the indices of the support patches. An illustration is given in the image above. We can select any of the change types used by the SECOND dataset, so "low vegetation", "n.v.g. surface" (which means non-vegetated), "tree", "water", "building" or "playground" (which refers mostly to sports stadiums). Change will be restricted to pixels where a) some change occured, b) the label for the prechange image is one of the labels in `change_types_from` and c) the label for the postchange image is one of the labels in `change_types_to`.

Note that there are some special cases: One specialty of the SECOND dataset is that changes can occur between the same change type, e.g., when one building is replaced by another. If we use the first few-shot task as an example, where we have `change_types_from: [n.v.g. surface, low vegetation]` and `change_types_to: [n.v.g. surface, low vegetation]`, this includes all of the following change types: n.v.g. surface to low vegetation, n.v.g. surface to n.v.g. surface, low vegetation to n.v.g. surface and low vegetation to low vegetation. Also, if we set one of the lists to `null`, as is done for change types two, three and four, it means that the labels in that time step can be anything, i.e., there is no restriction placed. This way, we can, e.g., define building construction as "from anything (`null`) to building).

The support patches are defined there as well, using the index from the support dataset (see *Indexing* below) with a patch size of 256 and a stride of 256, resulting in 4 patches per image. This is therefore not the same as the train dataloader that has 9 patches per image. The order here is also what matters for the amount of support patches: If we set the number of support patches to, e.g., 3 in the test script, then the first 3 patches that are listed for the task in the config.yaml are used.

## Indexing

There are several different ways to refer to individual images by number. In order to provide some clarity, we define them here and indicate where in the code we use them

- **File Name**: The number used in the file names of the SECOND dataset. For example, the very first image is called `00003.png`. The advantage of this numbering is that it is stable even when we delete or add files to the folders. We use this for the filenames of test images (see *Images* below).

- **File Number**: The index of the file when sorting alphanumerically, starting with 0. This is used by the `dataset` class to define the train, test and validation splits, and to prevent overlap (see *Overlap* below).
- **Dataset**: The index used by a dataset object, i.e., denoting the image returned by `dataset[idx]`. This is not only the file_number multiplied by the number of patches per image, but also starts at 0 for each subset (so, e.g., if the first image in the validation set will have the file number 800, its first patch can be accessed by a validation dataset with `dataset[0]`). This is the way used to define the patches for the few-shot tasks in the `config.yaml` (see *The Few-Shot Tasks* in *The config.yaml* above).

As noted above, apart from the file name, all other indices will change when adding or deleting files from the folders of the data set. This is therefore not possible without (possibly silently!) breaking basically everything. Do not do this!

## Overlap

Unfortunately, we only had access to the train set of the SECOND dataset. Therefore, we need to create our own train-test-val split from this data. We use (roughly) a 8:1:1 split, using the first 80% of images for training, the next 10% for validation and the final 10% for testing. This is not ideal, as we have no information on, e.g., the spatial distribution, and if the order in the folder is determined by different cities etc., we might have a bit of distribution shift here.

Also, there are some overlaps between images, and in order to prevent data leakage, we made an effort to detect these overlaps as good as possible, and have moved some of the images that would be part of the test or validation set to the trainset. This is all done automatically by the `dataset` class, but with the file number (see *Indexing* above), so it is important not to delete or add any images or other files to the dataset folder!

## Images

Images produced by the test scripts as well as logged during training are colorcoded as described in the paper: Black pixels are true negatives, white are true positives, blue are false negatives (change that is present but was not detected) and red are false positives (change that was detected but is not present).

The filenames of the images that are created during testing are using the file names of the dataset as names, and additionally the number of the patch (from `0` to `8` in the case of 9 patches). So for example, 'XXXX_3.png' is the result for the fourth patch of the pair of images that can be found under 'xxxx.png' in the im1 and im2 folders in the dataset.

## Caveats

This implementation specifically needs torchgeo in version `0.2.1`, as later versions use a refactored variant of the `FCSiamConc` model. This should not affect the general numerical results too much (at least relative to each other), but **will** break the code in this repository. Adapting it to the newer models is easily possible, but is not what was used for the experiments in the paper, which is why we stick with the old version for reproducibility.

# Citation

When using this in your own work, please cite our paper:

    @article{hermann2023filtering,
        title = {Filtering Specialized Change in a Few-Shot Setting},
        author={Hermann, Martin and Saha, Sudipan and Zhu, Xiao Xiang},
        journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
    }