import torch
import torch.nn.functional as F
from torch import nn
from torchgeo.models import FCSiamConc

from modules import BaseChangeDetectionModel


class ChangeDetectionModel(BaseChangeDetectionModel):
    '''
        Standard Change Detection implementation that wraps a
        model in its `backbone` argument, with some slight adaptions such as
        removing the final layer and replacing it by a 1x1 convolution layer.

        Also adapts the main processing slightly, i.e. stripping away
        the final layers, disabling dropout etc.

        Args:
            - backbone: the change detection backbone to be wrapped. Has
              to be an instance of `FCSiamConc`
            - change_weight: the positive weight used in the loss that will
              apply to changed instances (unchanged ones will have weight 0)
            - backbone_dropout: dropout rate to be used in the backbone. If set
              to 0.0, this will disable dropout.
            - change_decision_threshold: the value that decides how sure we
              have to be until we predict change, i.e., the minimum value that
              a prediction needs to pass to be considered as changed.
            - display_epoch_number: controls whether to print the current epoch
              number to the console after each epoch. Mainly to be used when
              disabling the progress bar of the trainer to still have some
              information about the state of the training.

        Shape:
            - Input: :math:`(B, 2, C, H, W)`, with B the batch size, C the number
              of channels, and :math:`(H, W)` the image dimensions. The second
              dimension is the number of timesteps, which is fixed at 2
            - Output: :math:`(B, 1, H, W)`

        Attributes:
            backbone: see above
            change_decision_threshold: see above
            change_decision_layer: the 1x1 convolution layer that makes the
                final decision
            display_epoch_number: see above
            embedding_dim: the number of output channels of the second to last
                layer (exists for historical reasons)
            prediction_mode: one of 'binary' or 'logits_change', deciding
                the output of `make_prediction`: for 'binary', we have a tensor
                with '1' in every changed and '0' in every unchanged position,
                for 'logits_change', we get the raw sigmoid values between 0 and 1
            weight: see above (under change_weight)
    '''

    def __init__(self,
        backbone,
        change_weight: float = 1.,
        backbone_dropout: float = 0.2,
        change_decision_threshold: float = 0.5,
        display_epoch_number: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        if isinstance(backbone, FCSiamConc):
            # strip away the final layer
            del backbone.decoder[-1][-1]
            # and remove the dropout from the new last layer
            del backbone.decoder[-1][-1][-1]
            self.embedding_dim = backbone.decoder[-1][-1][-3].out_channels
            self.backbone=backbone
        else:
            raise ValueError(f'Type "{type(backbone)}" not supported!')

        # unfortunately, torchgeo does not provide options to change the dropout
        # in its fully convolutional modules
        for module in self.backbone.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.p = backbone_dropout

        self.change_decision_threshold = change_decision_threshold

        self.register_buffer('weight', torch.FloatTensor([change_weight]))

        self.change_decision_layer = nn.Conv2d(self.embedding_dim, 1, 1)

        self.display_epoch_number = display_epoch_number


    def forward(self, x):
        x = self.backbone(x)
        y = self.change_decision_layer(x)
        return y

    def training_step(self, batch, batch_idx):
        step_result = self._overall_step(batch)
        for loss_name, loss in step_result['losses'].items():
            self.log(f'loss/{loss_name}/train', loss)
        return step_result['losses']['overall_loss']

    def on_train_epoch_end(self):
        if self.display_epoch_number:
            print(f'Finished epoch {self.current_epoch}!')

    def validation_step(self, batch, batch_idx):
        step_result = self._overall_step(batch)
        for loss_name, loss in step_result['losses'].items():
            self.log(f'loss/{loss_name}/val', loss)

    def test_step(self, batch, batch_idx):
        _, losses = self._overall_step(batch)
        for loss in losses:
            self.log(f'loss/{loss}/test', losses[loss])

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.make_prediction(x)
        return y_hat

    def make_prediction(self, x, mode=None):
        '''
            Predicts change for a given input batch.

            Args:
                - x: the batch that we want to predict for
                - mode: the prediction mode. For `binary`, the values will be either
                  0 or 1, for `logits_change`, it will return the logit outputs
                  of the model after a sigmoid, resulting in values between 0 and 1.
                If not provided, we use `self.prediction_mode`.

            Shape:
              - Input: :math:`(B, 2, C, H, W)`, with B the batch size, C the number
                of channels, and :math:`(H, W)` the image dimensions. The second
                dimension is the number of timesteps, which is fixed at 2
              - Output: :math:`(B, H, W)`
        '''
        if mode is None:
            mode = self.prediction_mode
        change_decision = torch.sigmoid(self(x).squeeze(1))

        if mode == 'binary':
            change = (change_decision > self.change_decision_threshold)
            y_hat = change.int()
        elif mode == 'logits_change':
            y_hat = change_decision
        else:
            raise ValueError(f'Unknown prediction mode {mode}!')
        return y_hat

    def _overall_step(self, batch):
        x, y = batch
        change_decision = self(x).squeeze(1)
        overall_loss = F.binary_cross_entropy_with_logits(
            change_decision,
            y.float(),
            pos_weight=self.weight,
        )
        return {
            'losses': {
                'overall_loss': overall_loss,
            },
        }
