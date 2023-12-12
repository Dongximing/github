from datasets import Dataset
from transformers import AutoTokenizer, pipeline
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
ds = Dataset.from_file("train_dataset/dataset.arrow")
import numpy as np
def tokenize(sample):
    # print(type(sample["prompt"]))
    # print("prompt :", sample["prompt"]["text"])
    sample["tokens"] = gpt2_tokenizer.encode(sample["prompt"]["text"])
    return sample
encoded_ds = ds.map(tokenize, batched=False)
encoded_ds = encoded_ds.map(lambda sample: {"length": len(sample["tokens"])}, batched=False)

# 找出最小长度
min_length = np.mean(encoded_ds['length'])
print("最小编码长度:", min_length)