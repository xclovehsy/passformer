"""Inference module for trained models."""

from .instbert_inference import InstBertInference
from .optseq_gen_inference import OptSeqGenInference

__all__ = ["InstBertInference", "OptSeqGenInference"]

