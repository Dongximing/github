
import time
import pandas as pd

from perspective import PerspectiveAPI
from tqdm import tqdm
API_KEY = 'AIzaSyD4-7vqsMGtYD19WVvtdUb5SKrgMI3OZ3k'
Perspective = PerspectiveAPI(API_KEY)

import time


def analyze_text(blocking_prompt, index):
    toxicity_score = None  # 初始设置为None，如果有错误，函数最终将返回这个值
    try:
        print(f"index = {index}")

        result = Perspective.score(blocking_prompt)
        toxicity_score = result.get("TOXICITY")  # 如果"TOXICITY"键不存在，这将返回None而不是引发异常
        print("Toxicity score: " + str(toxicity_score))

    except Exception as e:
        print(f"An error occurred: {e}")  # 打印错误消息

    finally:
        time.sleep(0.6)  # 确保在成功或失败的情况下都会暂停
        return toxicity_score  # 返回毒性评分，如果有错误或"TOXICITY"键不存在，则为None



df = pd.read_csv('paper_toxic_20000_training_steps.csv')

if "model_real_output" in df.columns:
    df['api_output_score'] = [analyze_text(text,index) for index, text in enumerate(df['model_real_output'])]

    average_score = df['api_output_score'].mean()

    print('api_output_score',average_score)



