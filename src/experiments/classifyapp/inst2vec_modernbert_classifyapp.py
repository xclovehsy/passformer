import os
import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from src.model.tokenizer import Inst2VecTokenizer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

def tokenize_func(examples, tokenizer, max_length=512):
    tokenized_data = tokenizer(
        examples['llvm'], 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    # 将 label 从 1-based 映射到 0-based  这里要求batch=false
    labels = int(examples['label']) - 1
    tokenized_data['labels'] = labels
    return tokenized_data

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if __name__ == '__main__':

    data_folder = '/home/xucong24/Compiler/datasets/POJ104Dataset'
    out_folder = '/home/xucong24/Compiler/work_dirs/modernbert_inst2vec_for_classifyapp'
    model_path = "/home/xucong24/Compiler/work_dirs/inst2vec_poj104_modernbert/20250825_063731/final_model"
    tokenizer_path = "/home/xucong24/Compiler/work_dirs/inst2vec_poj104_modernbert/20250825_063731/final_model"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    print("Loading tokenizer...")
    tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_path)
    
    print("Loading dataset...")
    dataset = datasets.load_from_disk(data_folder)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=False,
        num_proc=32,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': 512}  # 正确的语法
    )

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=104
    )

    print("Training...")
    training_args = TrainingArguments(
        output_dir=out_folder,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=out_folder,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['val'],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model(out_folder)
    tokenizer.save_pretrained(out_folder)

    

    
    
