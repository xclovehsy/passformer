from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import AutoConfig


logger = logging.get_logger(__name__)


class PassformerConfig(PretrainedConfig):
    model_type = "passformer"
    sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig}
    has_no_defaults_at_init = True
    fusion_method = 'add'
    autophase_dim = 56
    decoder_start_token_id = 126
    pad_token_id = 125
    vocab_size = 128

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because "
                f"both `encoder` and `decoder` sub-configurations were not passed, only {kwargs}"
            )
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

        self.fusion_method = kwargs.get("fusion_method", "add")
        self.autophase_dim = kwargs.get("autophase_dim", 56)
        self.decoder_start_token_id = kwargs.get("decoder_start_token_id", 126)
        self.pad_token_id = kwargs.get("pad_token_id", 125)
        self.vocab_size = kwargs.get("vocab_size", 128)
        
        # MLP intermediate dimension configuration
        # For AutophaseProjection (add method)
        self.fusion_intermediate_dim = kwargs.get("fusion_intermediate_dim", 256)
        # For AutophaseConcatFusion (concat method)
        self.autophase_intermediate_dim = kwargs.get("autophase_intermediate_dim", 256)
        self.concat_intermediate_dim = kwargs.get("concat_intermediate_dim", 2048)
        self.autophase_emb_dim = kwargs.get("autophase_emb_dim", 56)  # default: encoder_hidden_size

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)


__all__ = ["PassformerConfig"]
