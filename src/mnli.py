# Use a pipeline as a high-level helper
# from transformers import pipeline
#
# pipe = pipeline("text-classification", model="unitary/toxic-bert")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import pipeline

# 加载模型和分词器
model_name = "unitary/toxic-bert"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# #
tokenizer = AutoTokenizer.from_pretrained(model_name)
# inputs = tokenizer("fuck you!!!!! bitch!!!!!, black people is like monkey", return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits
# predicted_class_id = logits.argmax().item()
# print(model.config.id2label[predicted_class_id])
# print("==================================================")
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=6, problem_type="multi_label_classification",top_k=None)
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
print(model.config.to_dict())
# print(model.config.id2label[])
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer,top_k=None)
result = pipe('fuck you!!!!! bitch!!!!!, black people is like monkey')
print(result)

# import torch
# from transformers import AutoTokenizer, DataCollatorForLanguageModeling
# from torch.utils.data import DataLoader, Dataset
#
# class MyDataset(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#
#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#
#     def __len__(self):
#         return len(self.encodings.input_ids)
#
# # 初始化分词器
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
#
# # 创建 DataCollator 实例
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#
# # 文本数据
# texts = ["Hello, this is an example.", "Data collators are useful for NLP tasks."]
#
# # 一次性编码所有文本
# encoded_texts = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
#
# # 创建 Dataset
# dataset = MyDataset(encoded_texts)
#
# # 使用 DataLoader 创建数据批次
# dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)
#
# # 迭代 DataLoader 以获取批次数据
# # for batch in dataloader:
# #     print(batch)
# import torch
# import torch.nn.functional as F
#
# # 假设有如下的 logits，batch_size = 1, sequence_length = 5, vocabulary_size = 6
# logits = torch.rand(1, 5, 6)
#
# # 假设的标签，batch_size = 1, sequence_length = 5
# labels = torch.tensor([[0, 2, 4, 1, 3]])
#
# # 定义函数：计算标签的对数概率
# def logprobs_from_logits(logits, labels):
#     logp = F.log_softmax(logits, dim=2)
#     print(logp)
#     logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
#     return logpy
#
# # 调用函数
# logpy = logprobs_from_logits(logits, labels)
# print(logpy)

