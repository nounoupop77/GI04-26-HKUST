from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("custom")
@dataclass
class CustomConfig(PreTrainedConfig):
    """ResNet-18 + SpatialSoftmax + GRU + KeypointEncoder imitation policy."""

    fc_hidden_dim: int = 256
    state_dim: int = 6
    action_dim: int = 6
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-3
    encoder_lr: float = 5e-5

    num_frames: int = 3
    action_chunk_size: int = 8
    bn_train_in_layer4: bool = False

    # ----- block + target keypoint head -----
    keypoint_dim: int = 8          # (block_x, block_y, target_x, target_y)
    keypoint_emb_dim: int = 64
    # Probability that the WHOLE keypoint vector is zeroed during TRAINING.
    #   0.0  -> 100% of the detection is always used
    #   0.8  -> keypoint is zeroed 80% of the time (~20% kept)
    #   1.0  -> ablation: never use detection
    keypoint_dropout_prob: float = 0.2

    def __post_init__(self):
        super().__post_init__()

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("CustomPolicy requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("CustomPolicy requires 'action' in output_features.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None