from transformers import (
    EncoderDecoderConfig,
    AutoConfig,
    PretrainedConfig
)

class PassformerConfig(EncoderDecoderConfig):

    model_type = "passformer"

    def __init__(
        self,
        fusion_method: str = "add",  # "concat", "add", "cross_attention"
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fusion_method = fusion_method
        self.autophase_dim = 56
        self.decoder_start_token_id = 126
        self.pad_token_id = 125
        self.vocab_size = 128
