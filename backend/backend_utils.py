from datasets import Dataset, DatasetDict, load_from_disk
import os
import numpy as np
import pandas as pd

import torch

import datasets
from transformers import BitsAndBytesConfig

def _setHFToken():
    with open("../hf_token.txt", "r") as file:
        token = file.read()
        
    return token

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


# 4 bit Normal Form
# converting 32 bit to 4 bit
# to balance loss of information due to quantization we keep the new fine-tuned params in 16 bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_id = 'google/gemma-7b'
