"""Optimization Sequence Generation Inference."""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union

from ..model import (
    Inst2VecTokenizer, 
    OptiSeqTokenizer, 
    PassformerModel
)


class PassformerInference:

    def __init__(
        self,
        model: PassformerModel,
        encoder_tokenizer: Inst2VecTokenizer,
        decoder_tokenizer: OptiSeqTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.device = device
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        encoder_tokenizer_path: Optional[str] = None,
        decoder_tokenizer_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> "PassformerInference":

        # 加载模型
        model = PassformerModel.from_pretrained(model_path)
        
        # 确定 tokenizer 路径
        if encoder_tokenizer_path is None:
            encoder_tokenizer_path = os.path.join(model_path, "encoder_tokenizer")
            if not os.path.exists(encoder_tokenizer_path):
                # 尝试其他可能的路径
                encoder_tokenizer_path = os.path.join(model_path, "Inst2VecTokenizer")
        
        if decoder_tokenizer_path is None:
            decoder_tokenizer_path = os.path.join(model_path, "decoder_tokenizer")
            if not os.path.exists(decoder_tokenizer_path):
                # 尝试其他可能的路径
                decoder_tokenizer_path = os.path.join(model_path, "OptiSeqTokenizer")
        
        # 加载 tokenizer
        if not os.path.exists(encoder_tokenizer_path):
            raise ValueError(
                f"Encoder tokenizer not found at {encoder_tokenizer_path}. "
                "Please specify encoder_tokenizer_path explicitly."
            )
        if not os.path.exists(decoder_tokenizer_path):
            raise ValueError(
                f"Decoder tokenizer not found at {decoder_tokenizer_path}. "
                "Please specify decoder_tokenizer_path explicitly."
            )
        
        encoder_tokenizer = Inst2VecTokenizer.from_pretrained(encoder_tokenizer_path)
        decoder_tokenizer = OptiSeqTokenizer.from_pretrained(decoder_tokenizer_path)
        
        return cls(
            model=model,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            device=device
        )
    
    def _prepare_encoder_inputs(
        self,
        llvm: Union[str, List[str]],
        autophase: Optional[Union[List[List[float]], List[np.ndarray], torch.Tensor]] = None,
        max_input_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:

        # 检查模型是否需要 autophase
        requires_autophase = (
            hasattr(self.model.config, 'fusion_method') and 
            self.model.config.fusion_method is not None and 
            self.model.config.fusion_method != "none"
        )
        
        if requires_autophase and autophase is None:
            raise ValueError(
                f"Model requires autophase (fusion_method={self.model.config.fusion_method}), "
                "but autophase is None"
            )
        
        # 处理 LLVM IR tokenization
        is_batch = isinstance(llvm, list)
        if not is_batch:
            llvm = [llvm]
        
        # Tokenize LLVM IR
        encoded = self.encoder_tokenizer(
            llvm,
            max_length=max_input_length,
            padding=True,
            truncation=max_input_length is not None,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # 处理 autophase
        if autophase is None:
            # 如果不需要 autophase，返回 None
            autophase_tensor = None
        else:
            if isinstance(autophase, torch.Tensor):
                autophase_tensor = autophase
                if autophase_tensor.dim() == 1:
                    # [56] -> [1, 56]
                    autophase_tensor = autophase_tensor.unsqueeze(0)
            elif isinstance(autophase, (list, np.ndarray)):
                # 转换为 tensor
                if isinstance(autophase, np.ndarray):
                    if autophase.ndim == 1:
                        autophase = [autophase.tolist()]
                    else:
                        autophase = autophase.tolist()
                
                # 确保是列表的列表
                if len(autophase) > 0 and not isinstance(autophase[0], (list, np.ndarray)):
                    autophase = [autophase]
                
                # 转换为 tensor
                autophase_tensor = torch.tensor(autophase, dtype=torch.float32)
            else:
                raise TypeError(
                    f"Unsupported autophase type: {type(autophase)}. "
                    "Expected torch.Tensor, List[List[float]], List[np.ndarray], or None"
                )
            
            # 确保 batch size 匹配
            batch_size = input_ids.size(0)
            if autophase_tensor.size(0) == 1 and batch_size > 1:
                # 广播单个 autophase 到所有样本
                autophase_tensor = autophase_tensor.expand(batch_size, -1)
            elif autophase_tensor.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: llvm has {batch_size} samples, "
                    f"but autophase has {autophase_tensor.size(0)} samples"
                )
            
            # 验证 autophase 维度
            if autophase_tensor.size(1) != self.model.config.autophase_dim:
                raise ValueError(
                    f"Autophase dimension mismatch: expected {self.model.config.autophase_dim}, "
                    f"got {autophase_tensor.size(1)}"
                )
            
            autophase_tensor = autophase_tensor.to(self.device)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if autophase_tensor is not None:
            result["autophase"] = autophase_tensor
        
        return result
    
    def generate(
        self,
        llvm: Union[str, List[str]],
        autophase: Optional[Union[List[List[float]], List[np.ndarray], torch.Tensor]] = None,
        max_input_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_tensors: bool = False,
        **kwargs,
    ) -> Union[str, List[str], torch.Tensor]:

        # 准备输入
        encoder_inputs = self._prepare_encoder_inputs(
            llvm=llvm,
            autophase=autophase,
            max_input_length=max_input_length
        )
        
        # 设置生成参数
        generation_config = {
            "max_length": max_output_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "pad_token_id": self.decoder_tokenizer.pad_token_id,
            "eos_token_id": self.decoder_tokenizer.eos_token_id,
            "bos_token_id": self.decoder_tokenizer.bos_token_id,
        }
        
        # 仅在采样时添加温度相关参数
        if do_sample:
            generation_config["temperature"] = temperature
            if top_p < 1.0:
                generation_config["top_p"] = top_p
        
        # 移除 None 值
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        generation_config.update(kwargs)
        
        # 准备模型输入
        model_kwargs = {
            "input_ids": encoder_inputs["input_ids"],
            "attention_mask": encoder_inputs["attention_mask"],
        }
        if "autophase" in encoder_inputs:
            model_kwargs["autophase"] = encoder_inputs["autophase"]
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_kwargs,
                **generation_config
            )
        print("generated_ids:", generated_ids)
        # 解码
        if return_tensors:
            return generated_ids
        else:
            decoded = self.decoder_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            # 如果是单个输入，返回单个字符串
            if not isinstance(llvm, list):
                return decoded[0]
            return decoded


if __name__ == "__main__":
    
    # load model and tokenizer
    model_path = "D:/dev/passformer/checkpoints/passformer_test"
    encoder_tokenizer_path = "D:/dev/passformer/checkpoints/Inst2VecTokenizer"
    decoder_tokenizer_path = "D:/dev/passformer/checkpoints/OptiSeqTokenizer"
    inference = PassformerInference.from_pretrained(model_path, encoder_tokenizer_path, decoder_tokenizer_path)

    # test data
    with open("D:/dev/passformer/src/utils/qsort.ll", "r") as f:
        llvm = f.read()
    autophase = torch.randn(1, 56)
    print("generate:", inference.generate(llvm, autophase, do_sample=True))

    # test batch
    llvm = [llvm, llvm]
    autophase = torch.randn(2, 56)
    print("generate batch:", inference.generate(llvm, autophase, do_sample=True))