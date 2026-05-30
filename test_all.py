from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 模型路径（请确保路径与你下载的一致）
model_name = "/mnt/data/deepseek-llm-7b-chat" 

# 2. 准备 5 个测试问题
questions = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道",
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    """请解释以下对话中“意思”分别是什么意思：
    领导：你这是什么意思？
    小明：没什么意思。意思意思。
    领导：你这就不够意思了。
    小明：小意思，小意思。
    领导：你这人真有意思。
    小明：其实也没有别的意思。
    领导：那我就不好意思了。
    小明：是我不好意思。"""
]

print("正在加载模型，请稍候...")

# 3. 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    device_map="auto", 
    torch_dtype="auto"
).eval()

print("\n" + "="*50)
print("开始进行大模型语义理解测试")
print("="*50 + "\n")

# 4. 循环运行每个问题
for i, prompt in enumerate(questions, 1):
    print(f"测试问题 {i}: {prompt.strip()}")
    print("-" * 20 + " 模型回答 " + "-" * 20)
    
    # 编码并将输入移至 GPU
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    # 生成回答
    outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_p=0.9)
    
    # 解码并打印输出
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(response.strip())
    print("\n" + "="*50 + "\n")

print("测试完成！")