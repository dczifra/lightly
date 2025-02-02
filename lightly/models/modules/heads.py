""" Projection and Prediction Heads for Self-supervised Learning """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List, Tuple

import torch
import torch.nn as nn

from lightly.models import utils


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])

    """

    def __init__(self, blocks: List[Tuple[int, int, nn.Module, nn.Module]]):

        super(ProjectionHead, self).__init__()

        self.layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            self.layers.append(nn.Linear(input_dim, output_dim))
            if batch_norm:
                self.layers.append(batch_norm)
            if non_linearity:
                self.layers.append(non_linearity)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.

        """
        return self.layers(x)


class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins.

    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]

    [0]: 2021, Barlow Twins, https://arxiv.org/abs/2103.03230

    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(BarlowTwinsProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class BYOLProjectionHead(ProjectionHead):
    """Projection head used for BYOL.

    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]

    [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(BYOLProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class MoCoProjectionHead(ProjectionHead):
    """Projection head used for MoCo.

    "(...) we replace the fc head in MoCo with a 2-layer MLP head (hidden layer
    2048-d, with ReLU)" [0]

    [0]: MoCo, 2020, https://arxiv.org/abs/1911.05722

    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(MoCoProjectionHead, self).__init__([
            (input_dim, hidden_dim, None, nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class NNCLRProjectionHead(ProjectionHead):
    """Projection head used for NNCLR.

    "The architectureof the projection MLP is 3 fully connected layers of sizes
    [2048,2048,d] where d is the embedding size used to apply the loss. We use
    d = 256 in the experiments unless otherwise stated. All fully-connected
    layers are followed by batch-normalization [36]. All the batch-norm layers
    except the last layer are followed by ReLU activation." [0]

    [0]: NNCLR, 2021, https://arxiv.org/abs/2104.14548

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(NNCLRProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, nn.BatchNorm1d(output_dim), None),
        ])


class NNCLRPredictionHead(ProjectionHead):
    """Prediction head used for NNCLR.

    "The architecture of the prediction MLP g is 2 fully-connected layers
    of size [4096,d]. The hidden layer of the prediction MLP is followed by
    batch-norm and ReLU. The last layer has no batch-norm or activation." [0]

    [0]: NNCLR, 2021, https://arxiv.org/abs/2104.14548

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(NNCLRPredictionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used for SimCLR.

    "We use a MLP with one hidden layer to obtain zi = g(h) = W_2 * σ(W_1 * h)
    where σ is a ReLU non-linearity." [0]

    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(SimCLRProjectionHead, self).__init__([
            (input_dim, hidden_dim, None, nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class SimSiamProjectionHead(ProjectionHead):
    """Projection head used for SimSiam.

    "The projection MLP (in f) has BN applied to each fully-connected (fc)
    layer, including its output fc. Its output fc has no ReLU. The hidden fc is
    2048-d. This MLP has 3 layers." [0]

    [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(SimSiamProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, nn.BatchNorm1d(output_dim), None),
        ])


class SimSiamPredictionHead(ProjectionHead):
    """Prediction head used for SimSiam.

    "The prediction MLP (h) has BN applied to its hidden fc layers. Its output
    fc does not have BN (...) or ReLU. This MLP has 2 layers." [0]

    [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(SimSiamPredictionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class SwaVProjectionHead(ProjectionHead):
    """Projection head used for SwaV.

    [0]: SwAV, 2020, https://arxiv.org/abs/2006.09882
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(SwaVProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class SwaVPrototypes(ProjectionHead):
    """Prototypes used for SwaV.

    Each output feature is assigned to a prototype, SwaV solves the swapped
    predicition problem where the features of one augmentation are used to
    predict the assigned prototypes of the other augmentation.

    Examples:
        >>> # use features with 128 dimensions and 512 prototypes
        >>> prototypes = SwaVPrototypes(128, 512)
        >>>
        >>> # pass batch through backbone and projection head.
        >>> features = model(x)
        >>> features = nn.functional.normalize(features, dim=1, p=2)
        >>>
        >>> # logits has shape bsz x 512
        >>> logits = prototypes(features)

    """
    def __init__(self,
                 input_dim: int,
                 n_prototypes: int):
        super(SwaVPrototypes, self).__init__([])
        self.layers = nn.Linear(input_dim, n_prototypes, bias=False)

    def normalize(self):
        """Normalizes the prototypes so that they are on the unit sphere."""
        utils.normalize_weight(self.layers.weight)


class DINOProjectionHead(ProjectionHead):
    """Projection head used in DINO.

    "The projection head consists of a 3-layer multi-layer perceptron (MLP) 
    with hidden dimension 2048 followed by l2 normalization and a weight
    normalized fully connected layer with K dimensions, which is similar to the
    design from SwAV [1]." [0]

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: SwAV, 2020, https://arxiv.org/abs/2006.09882

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        output_dim:
            The output dimension of the head.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        batch_norm:
            Whether to use batch norm or not. Should be set to False when using
            a vision transformer backbone.
    
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        batch_norm=False, 
    ):
        bn = nn.BatchNorm1d(hidden_dim) if batch_norm else None

        super().__init__([
            (input_dim, hidden_dim, bn, nn.GELU()),
            (hidden_dim, hidden_dim, bn, nn.GELU()),
            (hidden_dim, bottleneck_dim, None, None),
        ])
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, module):
        """Initializes layers with a truncated normal distribution.
        
        """
        if isinstance(module, nn.Linear):
            utils._no_grad_trunc_normal(
                module.weight, 
                mean=0, 
                std=0.2, 
                a=-2, 
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one forward pass through the head.
        
        """
        x = self.layers(x)
        # l2 normalization
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x