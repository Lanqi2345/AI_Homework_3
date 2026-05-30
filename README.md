# 大语言模型部署项目

项目链接https://github.com/Lanqi2345/AI_Homework_3.git

## 一、项目简介

### （一） 基本介绍

本项目基于阿里云魔搭（ModelScope）开源模型社区平台，利用其云计算资源与Jupyter Notebook开发环境，完成三款主流国产开源大语言模型的部署与对比评测实践。项目选取了DeepSeek-LLM-7B-Chat、通义千问Qwen-7B-Chat及智谱ChatGLM3-6B作为研究对象，这三款模型均为70亿参数级别左右的代表性对话模型，但在训练数据、架构设计及优化策略上各具特色。通过Git克隆官方代码仓库并依据各模型部署文档配置运行环境，项目成功实现了三个模型的加载与推理服务搭建，进行了从环境准备到模型可用的全流程部署。并针对一些问答测试对三个模型的表现进行横向对比分析。

### （二） 实验环境说明

- **计算平台**：ModelScope (魔搭社区) GPU实例。
- **硬件配置**：8核CPU/32GB内存/24GB显存。
- **操作系统**：Ubuntu 22.04
- **环境管理**：Miniconda3 + Python 3.10。
- **核心库**：transformers (4.33.3), torch (2.3.0), modelscope (1.9.5)。
- **测试模型**：
  - **Qwen-7B-Chat**
  - **ChatGLM3-6B**
  - **DeepSeek-LLM-7B-Chat**

### （三）测试任务

本项目选用五道经典的中文语义逻辑测试题，编写Python推理脚本，对上述三个模型进行测试。问题如下：

- 请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少
- 请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上
- 他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道
- 明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？
- 领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。

## 二、 模型部署方法

### （一）环境配置

#### 1. Conda环境安装

默认GPU镜像中未预装Conda环境，进行手动下载。

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

运行安装程序

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
```

刷新环境变量

```bash
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

检查版本

```bash
conda --version
```

#### 2. 创建并激活环境

```bash
conda create -n qwen_env python=3.10 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen_env
```

#### 3.安装依赖

```bash
pip install torch==2.3.0 torchvision==0.18.0 

pip install -U pip setuptools wheel

pip install \
"intel-extension-for-transformers==1.4.2" \
"neural-compressor==2.5" \
"transformers==4.33.3" \
"modelscope==1.9.5" \
"pydantic==1.10.13" \"sentencepiece" \
"tiktoken" \
"einops" \
"transformers_stream_generator" \
"uvicorn" \
"fastapi" \
"yacs" \
"setuptools_scm"


pip install fschat --use-pep517

pip install accelerate
```

### （二）模型下载

切换到数据目录

```bash
cd /mnt/data
```

#### 1. Qwen-7B-Chat模型

下载通义千问`Qwen-7B-Chat`模型

```bash
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
```

#### 2. ChatGLM3-6B模型

下载智谱`ChatGLM3-6B`模型

```bash
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
```

#### 3. DeepSeek-LLM-7B-Chat模型

下载`DeepSeek-LLM-7B-Chat`模型

```bash
git clone https://www.modelscope.cn/deepseek-ai/deepseek-llm-7b-chat.git
```

### （三）编写推理脚本

一共编写了三个推理脚本，`run_qwen_cpu.py`在cpu上进行逐个问题测试，`run_qwen_gpu.py`在gpu上进行逐个测试，`test_all.py`在gpu上一次性测试所有问题，脚本内容详见代码。

### （四）运行实例

切换工作目录

```bash
cd /mnt/workspace
```

运行

```bash
python run_qwen_gpu.py
python test_all.py
```

## 三、 大语言模型横向对比分析总结

| 评测维度   | Qwen-7B-Chat | ChatGLM3-6B | DeepSeek-LLM-7B |
| :--------- | :----------- | :---------- | :-------------- |
| 回答一致性 | 较低         | 极高        | 极高            |
| 语义辨析   | 卓越         | 一般        | 较差            |
| 多义词辨析 | 一般         | 较差        | 极差            |
| 句法分析   | 卓越         | 较差        | 较差            |
| 逻辑推理   | 失效         | 有效        | 失效            |

