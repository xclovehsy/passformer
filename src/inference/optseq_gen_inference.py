"""Optimization Sequence Generation Inference."""

import os
import torch
from typing import Dict, List, Optional, Union

from transformers import (
    EncoderDecoderModel,
    AutoTokenizer,
    GenerationConfig,
)

from src.model import Inst2VecTokenizer, OptiSeqTokenizer


class OptSeqGenInference:
    """优化序列生成推理类。
    
    用于 Encoder-Decoder 模型的推理，支持：
    - 序列生成（给定 LLVM IR 生成优化序列）
    - 获取 encoder 嵌入表示
    
    Example:
        >>> inferencer = OptSeqGenInference.from_pretrained("path/to/final_model")
        >>> output = inferencer.generate(llvm_ir)
        >>> print(output)
    """
    
    def __init__(
        self,
        model: EncoderDecoderModel,
        encoder_tokenizer,
        decoder_tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.device = device
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        encoder_tokenizer_type: str = "auto",
        decoder_tokenizer_type: str = "optiseq",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> "OptSeqGenInference":
        """从预训练路径加载模型和tokenizer。
        
        Args:
            model_path: 训练保存的模型路径 (work_dirs/.../final_model)
            encoder_tokenizer_type: encoder tokenizer 类型 ("auto" 或 "inst2vec")
            decoder_tokenizer_type: decoder tokenizer 类型 ("auto" 或 "optiseq")
            device: 推理设备
            
        Returns:
            OptSeqGenInference 实例
        """
        model = EncoderDecoderModel.from_pretrained(model_path)
        
        encoder_tokenizer_path = os.path.join(model_path, "encoder_tokenizer")
        
        if encoder_tokenizer_type == "inst2vec":
            encoder_tokenizer = Inst2VecTokenizer.from_pretrained(encoder_tokenizer_path)
        else:
            encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_path)
        
        decoder_tokenizer_path = os.path.join(model_path, "decoder_tokenizer")
        
        if decoder_tokenizer_type == "optiseq":
            decoder_tokenizer = OptiSeqTokenizer.from_pretrained(decoder_tokenizer_path)
        else:
            decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer_path)
        
        return cls(model, encoder_tokenizer, decoder_tokenizer, device)
    
    def _prepare_encoder_inputs(
        self,
        text: Union[str, List[str]],
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """准备 encoder 输入。"""
        inputs = self.encoder_tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def generate(
        self,
        text: Union[str, List[str]],
        max_input_length: int = 512,
        max_output_length: int = 128,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        early_stopping: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """生成优化序列。
        
        Args:
            text: 输入 LLVM IR
            max_input_length: encoder 最大输入长度
            max_output_length: decoder 最大生成长度
            num_beams: beam search 的 beam 数量
            num_return_sequences: 返回序列数量
            do_sample: 是否采样（False 为贪婪/beam search）
            temperature: 采样温度
            top_k: top-k 采样
            top_p: nucleus 采样
            early_stopping: 是否提前停止
            
        Returns:
            生成的优化序列
        """
        inputs = self._prepare_encoder_inputs(text, max_input_length)
        
        # 构建生成配置参数
        gen_config_kwargs = dict(
            max_length=max_output_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            **kwargs
        )
        # early_stopping 仅在 beam search (num_beams > 1) 时有效
        if num_beams > 1:
            gen_config_kwargs["early_stopping"] = early_stopping
        
        generation_config = GenerationConfig(**gen_config_kwargs)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                generation_config=generation_config
            )
        
        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        if isinstance(text, str) and num_return_sequences == 1:
            return decoded[0]
        
        return decoded
    
    def get_encoder_embeddings(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        pooling: str = "cls"
    ) -> Dict[str, torch.Tensor]:
        """获取 encoder 的嵌入表示。
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            pooling: 池化方式，"cls" 或 "mean"
            
        Returns:
            包含嵌入表示的字典
        """
        inputs = self._prepare_encoder_inputs(text, max_length)
        
        with torch.no_grad():
            encoder_outputs = self.model.encoder(**inputs)
            hidden_states = encoder_outputs.last_hidden_state
            
            if pooling == "cls":
                pooled = hidden_states[:, 0, :]
            elif pooling == "mean":
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
        
        return {
            "last_hidden_state": hidden_states,
            "pooled_embedding": pooled
        }
    
    def encode(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        pooling: str = "cls"
    ) -> torch.Tensor:
        """简化接口：直接返回 encoder 嵌入向量。"""
        return self.get_encoder_embeddings(text, max_length, pooling)["pooled_embedding"]
    
    def forward(
        self,
        text: Union[str, List[str]],
        labels: Optional[Union[str, List[str]]] = None,
        max_input_length: int = 512,
        max_output_length: int = 128
    ) -> Dict[str, torch.Tensor]:
        """前向传播，可用于计算 loss 或获取 logits。"""
        inputs = self._prepare_encoder_inputs(text, max_input_length)
        
        decoder_inputs = None
        if labels is not None:
            decoder_inputs = self.decoder_tokenizer(
                labels,
                max_length=max_output_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            decoder_inputs = {k: v.to(self.device) for k, v in decoder_inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=decoder_inputs["input_ids"] if decoder_inputs else None
            )
        
        return {
            "loss": outputs.loss if outputs.loss is not None else None,
            "logits": outputs.logits,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state
        }


def main():
    """命令行推理示例。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimization Sequence Generation Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input", type=str, required=True, help="Input file or string")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--encoder_tokenizer_type", type=str, default="auto", 
                        choices=["auto", "inst2vec"])
    parser.add_argument("--decoder_tokenizer_type", type=str, default="optiseq",
                        choices=["auto", "optiseq"])
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    inferencer = OptSeqGenInference.from_pretrained(
        args.model_path,
        encoder_tokenizer_type=args.encoder_tokenizer_type,
        decoder_tokenizer_type=args.decoder_tokenizer_type,
        device=args.device
    )
    
    if os.path.isfile(args.input):
        with open(args.input, "r") as f:
            text = f.read()
    else:
        text = args.input
    
    print("Generating optimization sequence...")
    output = inferencer.generate(
        text,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        num_beams=args.num_beams
    )
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

