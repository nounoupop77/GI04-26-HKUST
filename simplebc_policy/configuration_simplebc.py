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
"""Configuration for the SimpleBC policy: a pretrained ResNet backbone with an MLP regression head.

This is the "simplest viable policy" baseline for the FYP imitation learning study: it maps the
current visual observation and robot state directly to a single joint-space action, with no action
chunking, temporal ensembling, or latent-variable modeling.
"""
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("simplebc")
@dataclass
class SimpleBCConfig(PreTrainedConfig):
    """Configuration class for the SimpleBC policy.

    Inputs and outputs are inferred automatically from the dataset metadata by `make_policy`, so
    `input_features` and `output_features` normally stay at their empty defaults.

    Notes on the inputs and outputs:
        - At least one key starting with "observation.images." is required as an input (multiple
          keys are treated as multiple camera views, all passed through one shared backbone).
        - "observation.state" is optional but recommended as input.
        - "action" is required as the output key.

    Args:
        image_size: Resolution the camera images are resized to (square) before entering the backbone.
            Training and inference both apply it, keeping the two consistent. 224 matches the ResNet
            ImageNet pretraining protocol and cuts the convolution compute ~6x vs 480x640, enabling
            real-time CPU inference on the deployment machine. Must be a multiple of 32 (the ResNet
            downsampling factor). `None` keeps the dataset's native resolution.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a
            dilated convolution (yields a larger feature map, disabled by default for this baseline).
        mlp_hidden_dims: Hidden layer dimensions of the MLP regression head, in order. The final
            projection to the action dimension is appended automatically.
        dropout: Dropout probability applied after every hidden layer of the MLP head. 0.0 disables
            dropout, keeping the baseline as simple as possible.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE")
            to a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX).
        optimizer_lr: Learning rate for the MLP head parameters.
        optimizer_lr_backbone: Learning rate for the pretrained backbone parameters (kept smaller to
            avoid destroying the pretrained features early in training).
        optimizer_weight_decay: Weight decay for all parameters.
    """

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Vision backbone.
    image_size: int | None = 224
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # MLP regression head.
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.0

    # Training preset.
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"`dropout` must be in [0.0, 1.0). Got {self.dropout}.")
        if self.image_size is not None and self.image_size % 32 != 0:
            raise ValueError(
                f"`image_size` must be a multiple of 32 (the ResNet total downsampling factor). "
                f"Got {self.image_size}."
            )
        if not self.mlp_hidden_dims:
            raise ValueError("`mlp_hidden_dims` must contain at least one hidden layer dimension.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("SimpleBC requires at least one image input (e.g. observation.images.*).")
        if self.action_feature is None:
            raise ValueError("SimpleBC requires an 'action' output feature.")

    # Single-step behavior cloning: condition on the current observation only, predict the current
    # action only.
    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None
