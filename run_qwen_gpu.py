from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 配置路径和问题
model_name = "/mnt/data/deepseek-llm-7b-chat" # 本地路径
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"

# 2. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 3. 加载模型（使用 GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",    # 自动分配到 GPU
    torch_dtype="auto"    # 自动匹配精度
).eval()

# 4. 准备输入数据（关键：必须 .to("cuda") 搬运到显卡）
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# 5. 配置流式输出
streamer = TextStreamer(tokenizer)

# 6. 执行推理
print("\n--- 模型回答如下 ---\n")
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)