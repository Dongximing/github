import torch
from trl.gpt2 import GPT2HeadWithValueModel
# 假设 model1 和 model2 是两个 PyTorch 模型
model1 = GPT2HeadWithValueModel.from_pretrained('gpt2-medium')
model2 = GPT2HeadWithValueModel.from_pretrained('./model-gpt2-medium-full')
print("Model 1 structure:")
print(model1)
print("\nModel 2 structure:")
print(model2)
# 检查两个模型的参数是否相等
def check_model_equality(model1, model2):
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()

    for param_tensor in model1_state_dict:
        if param_tensor not in model2_state_dict:
            return False
        if not torch.equal(model1_state_dict[param_tensor], model2_state_dict[param_tensor]):
            return False

    return True

# 使用函数检查模型
are_models_equal = check_model_equality(model1, model2)
print(f"Models are {'equal' if are_models_equal else 'not equal'}")

model1_weights = model1.state_dict()
model2_weights = model2.state_dict()

# 比较权重
for key in model1_weights:
    if key in model2_weights:
        # 计算权重差异
        diff = torch.sum((model1_weights[key] - model2_weights[key]) ** 2)
        print(f"Difference in {key}: {diff.item()}")
    else:
        print(f"{key} is not present in both models.")