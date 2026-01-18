"""This module defines an API for processing LLVM-IR with inst2vec."""
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

from . import preprocess as inst2vec_preprocess
from .preprocess import GetStructTypes, PreprocessStatement


# Default resource paths relative to this module
_MODULE_DIR = Path(__file__).parent
_DEFAULT_VOCAB_PATH = _MODULE_DIR / "resources" / "dictionary.pickle"
_DEFAULT_EMBEDDING_PATH = _MODULE_DIR / "resources" / "embeddings.pickle"


class Inst2vecEncoder:
    """An LLVM encoder for inst2vec."""

    def __init__(self, vocab_path=None, embedding_path=None):
        """
        Initialize the Inst2vec encoder.
        
        Args:
            vocab_path: Path to the vocabulary pickle file. 
                       If None, uses the default bundled vocabulary.
            embedding_path: Path to the embeddings pickle file.
                           If None, uses the default bundled embeddings.
        """
        vocab_path = vocab_path or _DEFAULT_VOCAB_PATH
        embedding_path = embedding_path or _DEFAULT_EMBEDDING_PATH
        
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        with open(embedding_path, "rb") as f:
            self.embeddings = pickle.load(f)

        self.unknown_vocab_element = self.vocab["!UNK"]

    def preprocess(self, ir: str) -> List[str]:
        """Produce a list of pre-processed statements from an IR.
        
        Args:
            ir: Either a string of LLVM IR or a path to a file containing LLVM IR.
            
        Returns:
            A list of preprocessed statement strings.
        """
        if os.path.isfile(ir):
            with open(ir, "r") as f:
                ir = f.read()
            
        lines = [[x] for x in ir.split("\n")]
        try:
            structs = GetStructTypes(ir)
            for line in lines:
                for struct, definition in structs.items():
                    line[0] = line[0].replace(struct, definition)
        except ValueError:
            pass

        preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
        preprocessed_texts = [
            PreprocessStatement(x[0]) if len(x) else ""
            for x in preprocessed_lines
        ]
        return [x for x in preprocessed_texts if x]

    def encode(self, preprocessed: List[str]) -> List[int]:
        """Produce embedding indices for a list of pre-processed statements.
        
        Args:
            preprocessed: A list of preprocessed statement strings.
            
        Returns:
            A list of vocabulary indices.
        """
        return [
            self.vocab.get(statement, self.unknown_vocab_element)
            for statement in preprocessed
        ]

    def embed(self, encoded: List[int]) -> np.ndarray:
        """Produce a matrix of embeddings from a list of encoded statements.
        
        Args:
            encoded: A list of vocabulary indices.
            
        Returns:
            A numpy array of shape (num_statements, embedding_dim).
        """
        return np.vstack([self.embeddings[index] for index in encoded])


if __name__ == "__main__":
    llvm_ir = r"""
; ModuleID = 'example.bc'
source_filename = "example.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"Hello, World!\00", align 1

define i32 @compare(i8* %0, i8* %1) #0 {
  %3 = alloca i8*, align 8

  ; Comparing the two pointers (simple equality check)
  %4 = icmp eq i8* %0, %1   ; compare if %0 == %1
  %5 = zext i1 %4 to i32    ; convert boolean to integer (0 or 1)
  ret i32 %5
}

declare i32 @printf(i8* nocapture readonly) #1
"""
    
    encoder = Inst2vecEncoder()
    text = encoder.preprocess(llvm_ir)
    encode_text = encoder.encode(text)
    embed_text = encoder.embed(encode_text)
    
    print("Preprocessed:", text)
    print("Encoded:", encode_text)
    print("Embeddings shape:", embed_text.shape)

