import torch
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

from datasets import load_dataset,concatenate_datasets

from transformers import AutoTokenizer, pipeline

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch

config = {
    "model_name": "gpt2-medium",
    "cls_model_name": "lvwerra/distilbert-imdb",
    "steps": 40000,
    "batch_size": 128,
    "forward_batch_size": 128,
    "ppo_epochs": 4,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":0.99,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

experiment_name = 'model-gpt2-medium-1127'

# load imdb with datasets
ds = load_dataset('imdb', split='train')

# 分别筛选出正面和负面评价
pos_reviews = ds.filter(lambda x: x['label'] == 1)
neg_reviews = ds.filter(lambda x: x['label'] == 0)
#
# # 从每个类别中选择 1250 个样本
pos_reviews = pos_reviews.select(range(12480))
neg_reviews = neg_reviews.select(range(12480))

#合并这两个子集
ds = concatenate_datasets([pos_reviews, neg_reviews])
#ds = pos_reviews

ds = ds.rename_columns({'text': 'review'})
print('len',len(ds))

device0 = torch.device("cuda:0")
#device1 = torch.device("cuda:1")

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": config["forward_batch_size"]
}

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])

gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_model.to(device0)
gpt2_model_ref.to(device0)
sentiment_pipe = pipeline("sentiment-analysis",config['cls_model_name'],return_all_scores=True, device=0) # reward

# Freezing LM
for module in [gpt2_model.transformer, gpt2_model.lm_head]:
    for param in module.parameters():
        param.requires_grad = False
trainable_parameters = {name: param.requires_grad for name, param in gpt2_model.named_parameters() if param.requires_grad}
print(trainable_parameters)
input_size = 32
initial_params = {name: param.clone() for name, param in gpt2_model.named_parameters() if param.requires_grad}


def tokenize(sample):
    sample["tokens"] = gpt2_tokenizer.encode(sample["review"])[:input_size]
    sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
    return sample

ds = ds.map(tokenize, batched=False)


gen_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
    "max_length": 57,
    "temperature": 1
}

def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

dataloader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], collate_fn=collater, shuffle=True)
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)
total_ppo_epochs = int(np.ceil(config["steps"]/config['batch_size']))

for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
    print()
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device0) for t in batch["tokens"]]
    print("query_tensors",len(query_tensors))

    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(config['batch_size']):
        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0), **gen_kwargs)
        response_tensors.append(response.squeeze())
    batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time()-t

    #### Compute sentiment score
    t = time.time()
    texts = batch['response']

    print(texts[0])
    print(' ')

    # print(texts)

    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    print(pipe_outputs)
    rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device0)
    timing['time/get_sentiment_preds'] = time.time()-t

    print(torch.mean(rewards))
    print(' ')
    
    #### Run PPO step 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time()-t
     
    #### Log everything
    timing['time/epoch'] = time.time()-t0
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()

for name, param in gpt2_model.named_parameters():
    if param.requires_grad:
        change = torch.sum(initial_params[name] != param).item() > 0
        print(f"{name} changed: {change}")
os.makedirs(experiment_name)
gpt2_model.save_pretrained(experiment_name)
gpt2_tokenizer.save_pretrained(experiment_name)
