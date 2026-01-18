import torch
from transformers import DataCollatorForLanguageModeling
from src.model import Inst2VecTokenizer

tokenizer_id = "checkpoints/Inst2VecTokenizer"

# 初始化 tokenizer
tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_id)

# ====== 2. 编码解码测试 ======
with open('tmp/37902.ll', 'r') as f:
        llvm = f.read()
encoded = tokenizer(llvm, max_length=100, padding=True)
print("Encoded:", encoded)

decoded = tokenizer.decode(encoded["input_ids"])
print("Decoded:", decoded)

# ====== 3. 测试 DataCollatorForLanguageModeling ======
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.5  # 提高概率，方便看到效果
)

batch = [encoded, encoded]  # 两条一样的数据

print("\nOriginal batch:")
print(batch)

masked_batch = collator(batch)
print("\nMasked input_ids:")
print(masked_batch["input_ids"])
print("Labels (=-100 means ignored):")
print(masked_batch["labels"])
