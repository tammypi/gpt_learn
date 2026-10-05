# gpt_learn

跟随《Build a Large Language Model (from Scratch)》实现 GPT 的学习项目。

## 环境准备

需要 Python ≥ 3.10，推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

```bash
# 安装 uv（如已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 如果很慢，换成
pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 克隆本仓库
git clone https://github.com/tammypi/gpt_learn.git
cd gpt_learn

# 安装依赖（自动创建 .venv）
uv sync
```

## 运行示例

代码是顶层直接执行的脚本，需先进入 `src/gpt_learn` 目录：

```bash
cd src/gpt_learn

# 多头注意力（MultiHeadAttention 封装版）
python attention.py

# 多头注意力（Wrapper 版本，含简化自注意力 CasualAttention）
python attention_simple.py

# 分词与 Embedding（读取 ./the-verdict.txt）
python tokenizer.py

# GPT 模型骨架 + 文本生成
python gpt.py
```

## 项目结构

```
gpt_learn/
├── pyproject.toml      # 依赖声明（tiktoken、torch）
├── uv.lock             # 依赖锁定
├── src/
│   └── gpt_learn/
│       ├── __init__.py
│       ├── attention_simple.py   # 简化自注意力 + MHA Wrapper
│       ├── attention.py          # MultiHeadAttention（生产版）
│       ├── tokenizer.py          # 分词 + Embedding
│       ├── gpt.py                # GPT 模型 + 文本生成
│       └── the-verdict.txt       # 训练语料（《The Verdict》）
└── README.md
```

## 参考

- Sebastian Raschka, *Build a Large Language Model (from Scratch)*
