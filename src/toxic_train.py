import torch
import time
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import Dataset
tqdm.pandas()

from datasets import load_dataset, concatenate_datasets

from transformers import AutoTokenizer, pipeline

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch

config = {
    "model_name": "gpt2-large",
    "cls_model_name": "unitary/toxic-bert",
    "steps": 40000,
    "batch_size": 1,
    "forward_batch_size": 1,
    "ppo_epochs": 4,
    "lr": 1.41e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 0.99,
    "lam": 0.95,
    "cliprange": .2,
    "cliprange_value": .2,
    "vf_coef": .1,
}

experiment_name = 'toxic-1211-experiment'


device0 = torch.device("cuda:0")


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
sentiment_pipe = pipeline("text-classification", config['cls_model_name'], return_all_scores=True,  device=0)  # reward

# Freezing LM
for module in [gpt2_model.transformer, gpt2_model.lm_head]:
    for param in module.parameters():
        param.requires_grad = False
trainable_parameters = {name: param.requires_grad for name, param in gpt2_model.named_parameters() if
                        param.requires_grad}
print(trainable_parameters)

initial_params = {name: param.clone() for name, param in gpt2_model.named_parameters() if param.requires_grad}

ds = Dataset.from_file("dataset/train_dataset/dataset.arrow")

def tokenize(sample):
    # print(type(sample["prompt"]))
    # print("prompt :", sample["prompt"]["text"])
    sample["tokens"] = gpt2_tokenizer.encode(sample["prompt"]["text"])
    return sample


ds = ds.map(tokenize, batch_size=False)
gen_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
    "max_length": 20,
    "temperature": 2.0
}


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


dataloader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], collate_fn=collater, shuffle=True)
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)
total_ppo_epochs = int(np.ceil(config["steps"] / config['batch_size']))

for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):

    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device0) for t in batch["tokens"]]
    print("query_tensors", len(query_tensors))

    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(config['batch_size']):
        print("prompt:",gpt2_tokenizer.decode(query_tensors[i].unsqueeze(dim=0)))
        gen_kwargs['max_length'] = len(query_tensors[i])+20
        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),**gen_kwargs)
        response_tensors.append(response.squeeze()[len(query_tensors[i]):])
    batch['full'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]


    timing['time/get_response'] = time.time() - t

    #### Compute sentiment score
    t = time.time()
    texts = batch['full']

    print(texts[0])
    print(' ')
    print(texts[0])

    # print(texts)

    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    print(pipe_outputs)
    rewards = torch.tensor([1-output[0]["score"] for output in pipe_outputs]).to(device0)
    timing['time/get_sentiment_preds'] = time.time() - t

    print(torch.mean(rewards))
    print(' ')

    #### Run PPO step
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time() - t

    #### Log everything
    timing['time/epoch'] = time.time() - t0
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
