# Deprecated: This module has been moved to src.core.inst2vec
# Keeping this for backward compatibility
from src.core.inst2vec import (
    preprocess,
    PreprocessStatement,
    GetStructTypes,
    GetFunctionsDeclaredInFile,
    Inst2vecEncoder,
    Inst2VecTokenizer,
)

__all__ = [
    "preprocess",
    "PreprocessStatement",
    "GetStructTypes",
    "GetFunctionsDeclaredInFile",
    "Inst2vecEncoder",
    "Inst2VecTokenizer",
]

import warnings
warnings.warn(
    "src.observation.inst2vec is deprecated. Use src.core.inst2vec instead.",
    DeprecationWarning,
    stacklevel=2
)
