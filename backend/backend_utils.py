from datasets import Dataset, DatasetDict, load_from_disk
import os
import numpy as np
import pandas as pd
from pprint import pprint

import torch
import wandb

import datasets
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments, logging, pipeline,EarlyStoppingCallback,IntervalStrategy)

# MONKEY PATCHIN
# def new_num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = True) -> int:
#     total_numel = []
#     for name, param in self.named_parameters():
#         if not only_trainable or param.requires_grad:
#             if exclude_embeddings and name.startswith("base_model."):
#                 continue
#             quant_storage = param.storage().cpu().numpy().dtype if param.is_quantized else param.storage().cpu().numpy().dtype
#             nb_params = quant_storage.itemsize
#             total_numel.append(param.numel() * nb_params)
#     return sum(total_numel)

# ModuleUtilsMixin.num_parameters = new_num_parameters

#################

def spiltDataset(dataset, train_ratio=0.8, test_ratio=0.1, seed=42):
    dataset = dataset.shuffle(seed=seed)
    
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_test = int(test_ratio * num_samples)
    num_valid = num_samples - num_train - num_test
    
    train_dataset = dataset.select(range(num_train))
    test_dataset = dataset.select(range(num_train, num_train + num_test))
    valid_dataset = dataset.select(range(num_valid))
    
    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "valid": valid_dataset
    })

# tokenz = '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'
os.environ['HF_TOKEN'] = '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'
os.environ['WANDB_TOKEN'] = 'b3310490fdea1283957046098c23956ef9606e32'

dataset = load_from_disk("../dataset")
dataset = spiltDataset(dataset)

pprint(dataset)

model_id = 'google/gemma-2b'
new_model_id = 'Therapy_Gemma_2b_QLoRA'
output_dir = "../results"
model_dir = "./model_dir"

epochs = 1
per_device_train_batch_size =1
per_device_eval_batch_size = 8
max_seq_length = 1024
## It says the effective batch size = per_device_train_batch_size * gradient_accumulation_steps, so we can increase the effective
# ##batch size without running out of memory
gradient_accumulation_steps=4
## It saves memory by checkpointing the gradients (set to True if memory is an issue)
gradient_checkpointing = False

# Hyperparameters set as recommended in the qLoRA paper, B.2 Hyperparameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
fp16 = True
bf16 = False

save_steps = 500
logging_steps = 10
eval_steps = 500

max_grad_norm = 0.3
learning_rate = 2e-5
weight_decay = 0.001
optim = "paged_adamw_8bit" ## paged optim to save memory 32bit or 8bit
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

if torch.cuda.is_available():
    device = torch.device(type='cuda', index=0)
    properties = torch.cuda.get_device_properties(device)
    print("Using CUDA device:", device)
    print("Total memory available:", properties.total_memory / (1024 * 1024), "MB")
else:
    device = torch.device(type='cpu', index=0)
    print("Using CPU device:", device)

# 4 bit Normal Form
# converting 32 bit to 4 bit
# to balance loss of information due to quantization we keep the new fine-tuned params in 16 bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    evaluation_strategy="steps",
    save_steps=save_steps,
    eval_steps=eval_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    resume_from_checkpoint=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim=optim,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map = device,
    token = os.environ['HF_TOKEN']
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token = os.environ['HF_TOKEN'])
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'right'
model.resize_token_embeddings(len(tokenizer))

wandb.login(key=os.environ['WANDB_TOKEN'])

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset['valid'],
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
)