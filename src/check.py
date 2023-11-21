import torch
from trl.gpt2 import GPT2HeadWithValueModel
# 假设 model1 和 model2 是两个 PyTorch 模型
model1 =GPT2HeadWithValueModel.from_pretrained('./model-gpt2-medium')
model2 = GPT2HeadWithValueModel.from_pretrained('./model-gpt2-medium-2560')

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