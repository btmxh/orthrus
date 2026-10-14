"""Config for Orthrus on the Qwen3.5 hybrid (attention + gated delta-net) backbone."""

from typing import Optional

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class OrthrusQwen3_5Config(Qwen3_5TextConfig):
    model_type = "orthrus_qwen3_5"

    def __init__(
        self,
        *args,
        block_size: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        mtp_seed_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        # Block slots 1..k pre-drafted by the frozen MTP head (0 = off, no MTP weights).
        self.mtp_seed_tokens = mtp_seed_tokens
