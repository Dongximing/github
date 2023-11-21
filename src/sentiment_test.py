import torch

import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
tqdm.pandas()

from datasets import load_dataset
import datasets

from transformers import GPT2Tokenizer, T5Tokenizer, BertTokenizer, BartForConditionalGeneration, PreTrainedTokenizerFast, BartTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, BertForSequenceClassification, GPT2LMHeadModel
from transformers import AutoTokenizer, pipeline, top_k_top_p_filtering
import torch.nn.functional as F
import torch

# from trl.gpt2 import GPT2HeadWithValueModel 
from trl.gpt2 import GPT2HeadWithValueModel, sentiment_generation
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch

# from evaluate import load
# from rouge_score import rouge_scorer, scoring
# from nltk.tokenize import sent_tokenize
# from distinct import distinct

config = {
    "model_name": "./model-gpt2-medium",
    "cls_model_name": "lvwerra/distilbert-imdb",
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1,
    "forward_batch_size":25
}

# load imdb with datasets

ds = pd.read_csv('/home/shaowei/sensitive-blocking/sentiment_methods/half_sentiment_context_imdb_paper_test_015.csv')
# ds = ds.rename_columns({'prompt': 'review'})
ds = ds.rename(columns={'prompt': 'review'})

# device0 = torch.device("cuda:0")
device1 = torch.device("cuda:0")

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": config["forward_batch_size"]
}

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])

gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model.to(device1)    
input_size = 8

def tokenize(sample):
    sample["tokens"] = gpt2_tokenizer.encode(sample["review"])[:input_size]
    sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
    return sample

ds = ds.apply(tokenize, axis=1)

bs = 25
result_data = dict()

query_tensors = ds['tokens'].tolist()
response_tensors = []


#### get response from gpt2 and gpt2_ref
with torch.no_grad():
    gpt2_model.eval()

    for i in tqdm(range(bs)):
        response = sentiment_generation(gpt2_model, torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device1))
        response_tensors.append(response[0])

#### decode responses
ds['model_real_output'] = [gpt2_tokenizer.decode(response_tensors[i]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q,r in zip(ds['review'], ds['model_real_output'])]
ds['completions'] = texts

save = ds.to_csv('paper.csv', index=False)
