#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SimpleBC policy: a pretrained ResNet backbone with an MLP regression head.

The plain behavior-cloning baseline for the FYP imitation learning study. Each camera image is
encoded by a shared (ImageNet-pretrained, BatchNorm-frozen) ResNet into a pooled feature vector;
the concatenated image features and robot state are regressed to a single joint-space action by a
small MLP. There is no action chunking and no internal state, so inference is a single forward
pass per control step.
"""

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_simplebc import SimpleBCConfig


class SimpleBCPolicy(PreTrainedPolicy):
    """Pretrained ResNet + MLP regression head behavior cloning policy."""

    config_class = SimpleBCConfig
    name = "simplebc"

    def __init__(
        self,
        config: SimpleBCConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default
                    instantiation of the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = SimpleBC(config)

        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        # Two parameter groups so the pretrained backbone fine-tunes with a smaller learning rate
        # than the freshly initialized MLP head.
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset. SimpleBC is stateless."""

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations.

        SimpleBC predicts a single action per invocation, so the chunk always has length 1.
        """
        self.eval()
        actions = self.model(self._group_images(batch))
        return actions.unsqueeze(1)  # (B, 1, action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        actions = self.predict_action_chunk(batch)
        return actions.squeeze(1)  # (B, action_dim)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        actions_hat = self.model(self._group_images(batch))

        # With `action_delta_indices == [0]` the dataset serves the action window; drop the
        # length-1 window dimension so both tensors are (B, action_dim).
        target_action = batch[ACTION]
        if target_action.ndim == 3:
            target_action = target_action[:, 0]
        action_is_pad = batch["action_is_pad"]
        if action_is_pad.ndim == 2:
            action_is_pad = action_is_pad[:, 0]

        abs_err = F.l1_loss(target_action, actions_hat, reduction="none")
        valid_mask = ~action_is_pad.unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        return l1_loss, {"l1_loss": l1_loss.item()}

    def _group_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch


class SimpleBC(nn.Module):
    """The underlying network: shared ResNet image encoder + MLP action regression head."""

    def __init__(self, config: SimpleBCConfig) -> None:
        super().__init__()
        self.config = config

        action_feature = config.action_feature
        if action_feature is None:
            raise ValueError(
                "SimpleBC requires an action output feature (FeatureType.ACTION) in `output_features`."
            )

        # Backbone for image feature extraction. BatchNorm is frozen (as in ACT) so the pretrained
        # normalization statistics survive training with small batches.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    config.replace_final_stride_with_dilation,
                ],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # `fc` would map the pooled features to the 1000 ImageNet classes; drop it and keep the
            # (B, backbone_dim) pooled feature vector.
            backbone_dim = backbone_model.fc.in_features
            backbone_model.fc = nn.Identity()
            self.backbone = backbone_model
        else:
            backbone_dim = 0

        # MLP head over [per-camera pooled image features, robot state].
        head_in_dim = backbone_dim * len(self.config.image_features)
        if self.config.robot_state_feature:
            head_in_dim += self.config.robot_state_feature.shape[0]

        dims = [head_in_dim, *self.config.mlp_hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(dims[-1], action_feature.shape[0]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns a (B, action_dim) batch of actions."""
        features: list[Tensor] = []
        if OBS_IMAGES in batch:
            for img in batch[OBS_IMAGES]:  # each (B, C, H, W)
                features.append(self.backbone(self._resize(img)))
        if self.config.robot_state_feature:
            features.append(batch[OBS_STATE])
        if not features:
            raise ValueError("SimpleBC received neither images nor robot state in the batch.")
        concatenated = torch.cat(features, dim=-1)
        return self.mlp(concatenated)

    def _resize(self, img: Tensor) -> Tensor:
        """Resize to `config.image_size` (square). Applied identically at train and inference time.

        The full field of view is kept (aspect ratio is squeezed rather than cropped), so objects at
        the workspace edges — the unseen-position robustness test cases — stay in frame.
        """
        size = self.config.image_size
        if size is None or (img.shape[-2] == size and img.shape[-1] == size):
            return img
        return F.interpolate(img, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
