from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch

class BaseChangeDetectionModel(pl.LightningModule, ABC):
    '''
        This is the base class for all change detection models.

        The main functionality at the moment is providing a unified optimizer
        to all its subclasses and the `make_prediction` interface.

        In this repository, there is currently only one subclass, namely
        `change_detection_model`, making this class somewhat redundant.
        It is mainly here because of historical reasons, but also provides
        nice separation between the optimizer arguments and the rest.

        Args:
            - lr: learning rate of the optimizer
            - lr_scheduler_type: one of 'step', 'multistep' or 'exponential,
              determining the type of learning rate scheduler
            - lr_scheduler_milestones: optional list of timesteps to be used
              with step or multistep learning rate schedulers
            - lr_scheduler_step_size: step size for step learning rate schedulers
            - lr_scheduler_gamma: gamma value to be used in the learning rate
              scheduler
            - prediction_mode: initial prediction mode, will be used by the
              implementation of `make_prediction`
            - weight_decay: value of weight decay to be used
    '''

    def __init__(self,
            lr = 0.001,
            lr_scheduler_type = 'exponential',
            lr_scheduler_milestones = None,
            lr_scheduler_step_size = 10,
            lr_scheduler_gamma = 0.95,
            prediction_mode = 'binary',
            weight_decay = 0.0,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_milestones = lr_scheduler_milestones
        self.lr_scheduler_step_size = lr_scheduler_step_size
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.prediction_mode = prediction_mode
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        if self.lr_scheduler_type== 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_scheduler_milestones,
                gamma=self.lr_scheduler_gamma,
            )
        elif self.lr_scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_scheduler_step_size,
                gamma=self.lr_scheduler_gamma,
            )
        elif self.lr_scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.lr_scheduler_gamma,
            )
        else:
            raise ValueError("Unknown LR Mode")

        return [optimizer], [scheduler]

    @abstractmethod
    def make_prediction(self, x, mode=None):
        '''
            Return a prediction for a single input batch `x`.

            This might be as simple as a forward pass followed by an argmax
            over all channels, but can theoretically also become quite complex.
        '''
