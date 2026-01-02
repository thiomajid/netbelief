from dataclasses import dataclass

from src.utils.types import ShardingRule


@dataclass(unsafe_hash=True, eq=True)
class LSTMForecasterConfig:
    num_metrics: int
    hidden_features: int
    num_heads: int = 4
    num_kv_heads: int = 2
    horizon: int = 1
    head_bias: bool = False
    normalize_qk: bool = True
    use_device_mixer: bool = True
    num_blocks: int = 3
    bidirectional: bool = False
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9, 0.95)
    attention_impl: str = "xla"
    conv_kernel_size: int = 3
    conv_bias: bool = False
    use_revin: bool = False

    def __post_init__(self):
        self.quantiles = tuple(self.quantiles)


@dataclass
class PatchLSTMConfig(LSTMForecasterConfig):
    patch_size: int = 16
    stride: int = 8
    pad_mode: str = "edge"
    pooling_mode: str = "last"

    def __post_init__(self):
        super().__post_init__()


@dataclass(unsafe_hash=True, eq=True)
class LSTMForecasterShardings:
    proj_conv_kernel: ShardingRule = (None, None, "tp")
    proj_conv_bias: ShardingRule = ("tp",)
    norm: ShardingRule = ("tp",)
    lstm_input_kernel: ShardingRule = (None, "tp")
    lstm_recurrent_kernel: ShardingRule = (None, "tp")
    lstm_bias: ShardingRule = ("tp",)
    attn_qkv: ShardingRule = (None, "tp")
    attn_output: ShardingRule = ("tp", None)
    attn_norm: ShardingRule = ("tp",)
    head_kernel: ShardingRule = ("tp", None)
    head_bias: ShardingRule = (None,)
