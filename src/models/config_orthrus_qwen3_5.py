"""Config for Orthrus on the Qwen3.5 hybrid (attention + gated delta-net) backbone."""

from typing import List, Optional

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class OrthrusQwen3_5Config(Qwen3_5TextConfig):
    model_type = "orthrus_qwen3_5"

    def __init__(
        self,
        *args,
        block_size: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        fixed_state_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        # Linear-attention layers whose diffusion twin reads the AR prefix state without
        # updating it (see OrthrusQwen3_5GatedDeltaNet).
        self.fixed_state_layers = list(fixed_state_layers or [])
