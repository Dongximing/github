
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("allenai/real-toxicity-prompts")

print(dataset)
train_test_split = dataset['train'].train_test_split(train_size=90000, test_size=9442)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

train_dataset.to_csv('train_dataset.csv')
test_dataset.to_csv('test_dataset.csv')

print("Train dataset and test dataset are saved as CSV files.")



