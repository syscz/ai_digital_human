#!/usr/bin/env bash
# AutoDL 实例一次性环境安装：ComfyUI + MiniMax H3 开源权重
# 用法：bash setup_h3.sh   （在解压后的 remote_bundle 目录里执行）
# 首次运行约 20~40 分钟（大头是 42GB 权重下载，AutoDL 内网/学术加速很快）
set -e

echo "==> [1/4] 安装 ComfyUI"
if [ ! -d ComfyUI ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI
fi
pip install -r ComfyUI/requirements.txt

echo "==> [2/4] 安装 H3 依赖（diffusers 管线备用）"
pip install diffusers transformers accelerate sentencepiece einops

echo "==> [3/4] 下载 MiniMax H3 模型组件（约 42GB，int8 剪枝版）"
# 三组件：扩散模型 / 文本编码器 / VAE，存放到 ComfyUI/models 对应目录
# 仓库地址可用环境变量覆盖（H3_MODEL_REPO），默认走 Comfy-Org 重打包
# （以官方文档实际仓名为准；国内网络建议 export HF_ENDPOINT=https://hf-mirror.com）
MODEL_REPO="${H3_MODEL_REPO:-Comfy-Org/MiniMax-H3_repackaged}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
pip install -q huggingface_hub
python download_h3_models.py --repo "$MODEL_REPO" --base ComfyUI/models

echo "==> [4/4] 校验"
python run_segments.py --selftest && echo "✅ 环境就绪。下一步：python run_segments.py"
