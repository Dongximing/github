
from tqdm import tqdm
from datasets import load_from_disk
import pandas as pd

tqdm.pandas()

from transformers import AutoTokenizer

import torch

from trl.gpt2 import GPT2HeadWithValueModel, sentiment_generation




config = {
    "model_name": "./toxic-1211-experiment",
    "cls_model_name": "unitary/toxic-bert",
    "forward_batch_size":2
}

# load imdb with datasets

ds = load_from_disk('/home/shaowei/sensitive-blocking/dataset/toxic_prompt_test')


device1 = torch.device("cuda:0")
output_file = 'paper_toxic_20000_training_steps.csv'


gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])

gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model.to(device1)
input_size = 8


bs = 2500
response_tensors = []
real_response_tensors = []
result_data = dict()
results = []
#### get response from gpt2 and gpt2_ref
with torch.no_grad():
    gpt2_model.eval()

    for i in tqdm(range(len(ds))):
        toxic_prompt = ds[i]['prompt']['text']
        query_tensors = gpt2_tokenizer.encode(toxic_prompt)
        print(query_tensors)
        input_size = len(query_tensors[0])
        response = sentiment_generation(gpt2_model, query_tensors.to(device1))


        results.append(
            {'prompt': toxic_prompt, 'model_real_output':gpt2_tokenizer.decode(response[0][input_size:]),"completions":gpt2_tokenizer.decode(response[0])})

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"All prompts have been processed and saved to {output_file}")



