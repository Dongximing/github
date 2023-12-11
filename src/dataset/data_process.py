
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("allenai/real-toxicity-prompts")

print(dataset)
train_test_split = dataset['train'].train_test_split(train_size=90000, test_size=9442)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

train_dataset.save_to_disk('train_dataset')
test_dataset.save_to_disk('test_dataset')

print("Train dataset and test dataset are saved as CSV files.")



