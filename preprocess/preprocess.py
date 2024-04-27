import pandas as pd
import unicodedata
import re
import os

import datasets
from datasets import list_datasets, load_dataset
from transformers import AutoTokenizer
from pprint import pprint

def _setHFToken():
    with open("../hf_token.txt", "r") as file:
        token = file.read()
        
    return token


tokenz = _setHFToken()
dataset = load_dataset("jerryjalapeno/nart-100k-synthetic", split="train")
pprint(dataset.info.__dict__)

SYSTEM_PROMPT = """You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
def preprocessText(text):
  text = re.sub(r'Alex', '', text)
  text = re.sub(r'Charlie', '', text)
  text = text.lower().strip()
  # remove ", " when it appears at the start of a sentence
  text = re.sub(r'^, ', '', text)
  # remove " ." with "."
  text = re.sub(r' \.', '.', text)
  # remove " ," with ","
  text = re.sub(r' ,', ',', text)
  # remove " ?" with "?"
  text = re.sub(r' \?', '?', text)
  # remove " !" with "!"
  text = re.sub(r' \!', '!', text)
  # remove ",." with "."
  text = re.sub(r',\.', '.', text)
  # remove ",?" with "?"
  text = re.sub(r',\?', '?', text)
  # remove more than one space
  text = re.sub(r' +', ' ', text)

  restext = ""
  for ch in unicodedata.normalize('NFD', text):
    if unicodedata.category(ch) != 'Mn':
      restext+=ch

  # restext = re.sub(r"([.!?])", r" \1", restext)
  restext = re.sub(r"[^a-zA-Z.!?]+", r" ", restext)

  short_forms = {
        r"\bi ve\b": "i have",
        r"\bi m\b": "i am",
        r"\bisn t\b": "is not",
        r"\baren t\b": "are not",
        r"\bwasn t\b": "was not",
        r"\bweren t\b": "were not",
        r"\bhaven t\b": "have not",
        r"\bhasn t\b": "has not",
        r"\bhadn t\b": "had not",
        r"\bwon t\b": "will not",
        r"\bwouldn t\b": "would not",
        r"\bdon t\b": "do not",
        r"\bdoesn t\b": "does not",
        r"\bdidn t\b": "did not",
        r"\bcan t\b": "cannot",
        r"\bcouldn t\b": "could not",
        r"\bshouldn t\b": "should not",
        r"\bmightn t\b": "might not",
        r"\bmustn t\b": "must not",
        r"\bain t\b": "am not"
    }

  for short_form, full_form in short_forms.items():
      restext = re.sub(short_form, full_form, restext)

  return restext.strip()


def preprocessDataset(row):
  id = row['id']
  row = row['conversations']
  for conversation in row:
    if conversation.get('from') == 'human':
      conversation['role'] = "human"
    elif conversation.get('from') == 'gpt':
      conversation['role'] = "assistant"

    conversation['content'] = preprocessText(conversation.get('value'))
    del conversation['from']
    del conversation['value']

  sys_dict = {
        'role': "system",
        'content': SYSTEM_PROMPT
    }
  row.insert(0, sys_dict)
  # Conversational format: messages
  return {"messages":row}

dataset = dataset.map(preprocessDataset, remove_columns=['conversations'])

tokenizer = AutoTokenizer.from_pretrained("philschmid/gemma-tokenizer-chatml", token=tokenz)
tokenizer.padding_side = 'right' 
original_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=tokenz)

# get special tokens
print(tokenizer.special_tokens_map)
print(original_tokenizer.special_tokens_map)

assert len(tokenizer) == len(original_tokenizer), "tokenizer are not having the same length"

# remove conversation with more than 1024 tokens, for training memory reasons.
dataset = dataset.map(lambda x: {"input_ids_length": len(tokenizer.apply_chat_template(x["messages"]))})
# filter out the samples that are too long
max_input_length = 1024
dataset = dataset.filter(lambda x: x["input_ids_length"] <= max_input_length)
dataset = dataset.remove_columns(["input_ids_length"])
print(dataset)

dataset.save_to_disk('../dataset')