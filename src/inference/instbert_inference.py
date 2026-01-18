import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from transformers import AutoModelForMaskedLM, AutoModel

from src.model import Inst2VecTokenizer


class InstBertInference:
    
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer: Inst2VecTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
    
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_mlm_head: bool = True
    ) -> "InstBertInference":

        tokenizer = Inst2VecTokenizer.from_pretrained(model_path)
        
        if load_mlm_head:
            model = AutoModelForMaskedLM.from_pretrained(model_path)
        else:
            model = AutoModel.from_pretrained(model_path)
        
        return cls(model, tokenizer, device)
    
    def _prepare_inputs(
        self, 
        llvm_ir: Union[str, List[str]], 
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:

        inputs = self.tokenizer(
            llvm_ir,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def get_embeddings(
        self, 
        llvm_ir: Union[str, List[str]], 
        max_length: int = 512,
        pooling: str = "cls"
    ) -> Dict[str, torch.Tensor]:

        inputs = self._prepare_inputs(llvm_ir, max_length)
        
        with torch.no_grad():
            if hasattr(self.model, 'base_model'):
                outputs = self.model.base_model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            hidden_states = outputs.last_hidden_state
            
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
    
    def mlm_predict(
        self, 
        llvm_ir: Union[str, List[str]], 
        max_length: int = 512,
        top_k: int = 5
    ) -> Dict[str, torch.Tensor]:

        if not hasattr(self.model, 'lm_head') and not hasattr(self.model, 'cls'):
            raise RuntimeError("Model does not have MLM head. Use load_mlm_head=True when loading.")
        
        inputs = self._prepare_inputs(llvm_ir, max_length)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_tokens = torch.topk(probs, k=top_k, dim=-1)
        
        return {
            "logits": logits,
            "top_k_tokens": top_k_tokens,
            "top_k_probs": top_k_probs
        }
    
    def fill_mask(
        self, 
        llvm_ir: str, 
        mask_positions: Optional[List[int]] = None,
        max_length: int = 512,
        top_k: int = 5
    ) -> List[Dict]:

        inputs = self._prepare_inputs(llvm_ir, max_length)
        input_ids = inputs["input_ids"]
        
        # 找到 MASK token 位置
        if mask_positions is None:
            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[1].tolist()
        
        if not mask_positions:
            return []
        
        results = self.mlm_predict(llvm_ir, max_length, top_k)
        
        predictions = []
        for pos in mask_positions:
            top_tokens = results["top_k_tokens"][0, pos].tolist()
            top_probs = results["top_k_probs"][0, pos].tolist()
            
            pred = {
                "position": pos,
                "predictions": [
                    {
                        "token": self.tokenizer._convert_id_to_token(tid),
                        "token_id": tid,
                        "probability": prob
                    }
                    for tid, prob in zip(top_tokens, top_probs)
                ]
            }
            predictions.append(pred)
        
        return predictions
    
    def encode(
        self, 
        llvm_ir: Union[str, List[str]], 
        max_length: int = 512,
        pooling: str = "cls"
    ) -> torch.Tensor:

        return self.get_embeddings(llvm_ir, max_length, pooling)["pooled_embedding"]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="InstBERT Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input", type=str, required=True, help="Input LLVM IR file or string")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    inferencer = InstBertInference.from_pretrained(args.model_path, device=args.device)
    
    import os
    if os.path.isfile(args.input):
        with open(args.input, "r") as f:
            llvm_ir = f.read()
    else:
        llvm_ir = args.input
    
    print("Computing embeddings...")
    embedding = inferencer.encode(llvm_ir, max_length=args.max_length, pooling=args.pooling)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[0, :10].tolist()}")


if __name__ == "__main__":
    main()

