"""Inference module for trained models."""

from .instbert_inference import InstBertInference
from .passformer_inference import PassformerInference

__all__ = ["InstBertInference", "PassformerInference"]

