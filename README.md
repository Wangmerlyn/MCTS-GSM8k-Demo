# MCTS-GSM8k-Demo
This is a repo for showcasing using Monte Carlo Search Tree(MCTS) with LLMs to solve gsm8k problems.

This is just a demo project that goes with my MCTS lecture video, so the implementation and code style can be a little bit rough, feel free to leave any issues or open PRs to contribute.

# MCTS code template

The MCTS implementation template is from [minimal MCTS](https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1).

# Explaination

Watch how I implemented this Monte Carlo Search Tree on [Bilibili(Chinese)](【手撕代码#2｜手写蒙特卡洛树搜索｜LLM和MCTS结合】 https://www.bilibili.com/video/BV1BArPYQE8x/?share_source=copy_web&vd_source=9b4d25b8767b0ea1804ade3ffd3cf9dc)

# Installation
```bash
pip install -r requirements.txt
```

# Run
## Running generation with GPT-4o
```bash
export OPENAI_API_KEY="Your API Key"
python main.py
```
