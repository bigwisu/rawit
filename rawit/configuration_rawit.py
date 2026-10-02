# Copyright 2025 Wisu Suntoyo
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

"""RawitConfig — PretrainedConfig for the Rawit decision model."""

from transformers import PretrainedConfig


class RawitConfig(PretrainedConfig):
    """Configuration for a Rawit decision model.

    Args:
        backbone_name: HuggingFace model id for the encoder backbone.
        num_head_layers: Number of Pre-LN Transformer layers in the decision head.
        head_hidden_dim: Hidden dimension of the decision head (must match backbone).
        dropout_prob: Dropout probability applied inside the decision head.
        temperature: Initial per-primitive temperature vector [choice, score, noul].
            Overwritten by post-hoc calibration; treat as a placeholder until calibrated.
        max_options: Maximum number of option markers supported ([OPT_0] .. [OPT_{n-1}]).
    """

    model_type = "rawit"

    def __init__(
        self,
        backbone_name: str = "aisingapore/SEA-LION-ModernBERT-300M",
        num_head_layers: int = 2,
        head_hidden_dim: int = 768,
        dropout_prob: float = 0.1,
        temperature: list = None,
        max_options: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.num_head_layers = num_head_layers
        self.head_hidden_dim = head_hidden_dim
        self.dropout_prob = dropout_prob
        # Three primitives: choice=0, score=1, noul=2
        self.temperature = temperature if temperature is not None else [1.0, 1.0, 1.0]
        self.max_options = max_options
