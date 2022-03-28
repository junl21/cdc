import copy

import torch
import torch.nn as nn


class CDC(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, num_classes, hidden_dim=2048, pred_dim=256):
        """
        hidden_dim: hidden feature dimension of projector and predictor (default: 2048)
        pred_dim: output feature dimension of projector and predictor (default: 256)
        """
        super(CDC, self).__init__()

        # create the online encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.online_encoder = base_encoder(num_classes=pred_dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.online_encoder.fc.weight.shape[1]
        self.online_encoder.fc = nn.Sequential(nn.Linear(prev_dim, hidden_dim, bias=False),
                                               nn.BatchNorm1d(hidden_dim),
                                               nn.ReLU(inplace=True),  # hidden layer
                                               nn.Linear(hidden_dim, pred_dim))  # output layer

        # create the target encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        # freeze target encoder
        for target_weight in self.target_encoder.parameters():
            target_weight.requires_grad = False

        # build a classifier
        self.fc = nn.Linear(pred_dim, num_classes)

    def forward(self, x):
        """
        Input:
            x: augmented views of images
        Output:
            output: classification results
            target_z: output features of target network
        """

        # compute features for input
        online_z = self.online_encoder(x)  # NxC

        with torch.no_grad():
            target_z = self.target_encoder(x)  # NxC

        return self.fc(online_z), target_z.detach()
