# Model module - Encoders and Tokenizers for LLVM IR processing
#
# This module provides:
# - Inst2vecEncoder: LLVM IR encoder using inst2vec embeddings
# - Inst2VecTokenizer: Tokenizer for LLVM IR (HuggingFace compatible)
# - OptiSeqTokenizer: Tokenizer for optimization sequences
# - PassformerModel: Encoder-Decoder with Autophase fusion

# from .inst2vec import (
#     Inst2vecEncoder,
#     preprocess,
#     PreprocessStatement,
#     GetStructTypes,
#     GetFunctionsDeclaredInFile,
# )
from .tokenizer import Inst2VecTokenizer, OptiSeqTokenizer
from .passformer.passformer import PassformerModel
from .passformer.passformer_config import PassformerConfig
# from .passformer import (
#     PassformerModel
# )
# from .passformer_config import PassformerConfig

__all__ = [
    # Encoders
    # "Inst2vecEncoder",
    # # Models
    "PassformerModel",
    "PassformerConfig",
    # "PassformerV2Model",
    # "PassformerV2Config",
    # Tokenizers
    "Inst2VecTokenizer",
    "OptiSeqTokenizer",
    # Data Collators
    # "PassformerDataCollator",
    # "PassformerV2DataCollator",
    # # Preprocessing functions
    # "preprocess",
    # "PreprocessStatement",
    # "GetStructTypes",
    # "GetFunctionsDeclaredInFile",
]
