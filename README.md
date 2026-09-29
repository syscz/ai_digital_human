# AI Digital Human

> 一个整合 **2D 面部动画** + **3D 全身数字人** 的 AI 数字人项目，覆盖从"图片说话"到"3D 全身跳舞"的完整技术栈。

---

## ✨ 功能一览

| 能力 | 技术方案 | 输入 | 输出 |
|------|---------|------|------|
| **2D 面部动画（视频驱动）** | LivePortrait（快手可灵） | 图片 + 驱动视频 | 带表情/头部/眨眼的说话视频 |
| **2D 面部动画（音频驱动）** | Wav2Lip | 图片 + 音频 | 嘴型同步视频 |
| **2D 端到端流水线** | Edge-TTS + SadTalker + Wav2Lip + GFPGAN | 图片 + 文字 | 带表情/头动的高清说话视频 MP4 |
| **3D 全身数字人** | Ready Player Me + Mixamo + Three.js | 3D Avatar + Mixamo 动作 | 浏览器实时渲染的 3D 数字人 |

---

## 🏗️ 项目结构

```
ai_digital_human/
├── digital_human_pipeline.py    ← 2D 端到端流水线（Edge-TTS → SadTalker → Wav2Lip → GFPGAN → 插帧）
├── 项目信息.md                    ← 详细技术文档（模型原理、硬件清单等）
│
├── LivePortrait/                ← 快手可灵出品，16.7k⭐（独立工具，未接入流水线）
│   ├── inference.py               命令行推理入口
│   ├── app.py                     Gradio 可视化界面
│   └── src/                       核心源码
│
├── SadTalker/                   ← OpenTalker/SadTalker，13k⭐（流水线的表情/头动生成）
│   ├── inference.py               推理入口（被流水线调用）
│   └── src/                       核心源码
│
├── Wav2Lip/                     ← Rudrabha/Wav2Lip，10k⭐（流水线的嘴型精修）
│   ├── inference.py               音频驱动推理入口
│   ├── models/                    Wav2Lip 模型定义
│   └── audio.py                   音频处理
│
├── gfpgan_weights/              ← GFPGANv1.4 人脸修复模型
├── gfpgan/weights/              ← facexlib 辅助模型（人脸检测/解析）
├── ffmpeg/                      ← 本地静态 ffmpeg（绕开 conda ffmpeg 的 DLL 冲突）
├── output/                      ← 流水线输出（音频 + 最终视频）
│
└── 3d_avatar/                   ← 3D 全身数字人（第一阶段）
    ├── index.html                 Three.js 前端
    ├── server.py                  Python 后端（Edge-TTS + 静态服务）
    ├── README.md                  子项目说明
    ├── 操作指引_ReadyPlayerMe_Mixamo.md
    ├── avatar/                    ← 放 Ready Player Me 的 .glb
    ├── animations/                ← 放 Mixamo 的 .fbx
    └── audio/                     ← TTS 输出目录
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU（可选，纯 CPU 也能跑但较慢）
- FFmpeg

### 1. 创建 conda 环境

```powershell
conda create -n LivePortrait python=3.10 -y
conda activate LivePortrait
```

### 2. 安装 PyTorch

```powershell
# CPU 版本
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 版本（有 NVIDIA GPU 时用）
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装依赖

```powershell
# LivePortrait 依赖
cd LivePortrait
pip install -r requirements.txt

# Wav2Lip 依赖
cd ../Wav2Lip
pip install -r requirements.txt

# Edge-TTS（语音合成）
pip install edge-tts

# FFmpeg（Windows）
conda install -c conda-forge ffmpeg -y
```

### 4. 下载预训练模型

```powershell
# LivePortrait 模型 (~628MB)
# 国内用户先设 HF 镜像
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"

# Wav2Lip 模型 (~415MB)
# 下载 wav2lip.pth 放到 Wav2Lip/checkpoints/
# 下载 s3fd.pth 放到 Wav2Lip/face_detection/detection/sfd/

# SadTalker 模型（流水线需要）
# 下载 SadTalker_V0.0.2_256.safetensors / SadTalker_V0.0.2_512.safetensors (~692MB each)、
#       mapping_00109-model.pth.tar (~149MB)、BFM_Fitting/ 整个目录 → SadTalker/checkpoints/
# 来源：ModelScope wwd123/sadtalker 或 GitHub OpenTalker/SadTalker 的 bag-of-bits release

# GFPGAN 模型（流水线需要）
# GFPGANv1.4.pth (~333MB) → gfpgan_weights/
# detection_Resnet50_Final.pth + parsing_parsenet.pth (facexlib) → gfpgan/weights/

# 详见 项目信息.md 的模型下载来源章节
```

### 5. 运行

#### 方式 A：LivePortrait Gradio（视频驱动面部动画）

```powershell
conda activate LivePortrait
cd LivePortrait
python app.py
# 浏览器打开 http://localhost:8890
```

#### 方式 B：Wav2Lip 命令行（音频驱动嘴型）

```powershell
conda activate LivePortrait
cd Wav2Lip
python inference.py --checkpoint_path checkpoints/wav2lip.pth --face your_photo.jpg --audio your_audio.wav --outfile result.mp4
```

#### 方式 C：端到端流水线（文字 → 说话视频）

```powershell
conda activate LivePortrait
cd d:\Work\Python\ai_digital_human

# 命令行（完整链路：SadTalker 表情/头动 → Wav2Lip 精修嘴型 → GFPGAN 修复 → 50fps 插帧）
python digital_human_pipeline.py --photo your_photo.jpg --text "你好，我是数字人" --voice zh-CN-XiaoxiaoNeural

# 快速链路（--no-sadtalker：只有嘴动，CPU 上快约一个量级）
python digital_human_pipeline.py --photo your_photo.jpg --text "你好" --no-sadtalker

# 其他常用开关
#   --sadtalker-size 256     SadTalker 用 256 渲染（更快，略不像本人）
#   --sadtalker-still        减少头部摆动（稳重口播风）
#   --no-gfpgan              关闭人脸修复
#   --no-interpolation       关闭 50fps 插帧
#   --keep-intermediate      保留 SadTalker 中间视频（默认成功后删除）

# Gradio 界面
python digital_human_pipeline.py --gradio
# 浏览器打开 http://localhost:7860
```

> ⏱ CPU 性能参考（无 NVIDIA GPU）：完整链路一段 15 秒语音约十几分钟（SadTalker 512 + 逐帧 GFPGAN 是大头）；`--no-sadtalker` 快速链路约 1-3 分钟。

#### 方式 D：3D 全身数字人（浏览器实时渲染）

```powershell
conda activate LivePortrait
cd 3d_avatar
python server.py
# 浏览器打开 http://localhost:8080
```

**3D 数字人需要准备素材**：
- Avatar: [Ready Player Me](https://readyplayer.me) → 导出 `.glb`（带 ARKit BlendShape）→ 放 `3d_avatar/avatar/`
- 动作: [Mixamo](https://www.mixamo.com) → 下载 `.fbx`（FBX Binary, With Skin）→ 放 `3d_avatar/animations/`

详细操作指引见 [3d_avatar/操作指引_ReadyPlayerMe_Mixamo.md](3d_avatar/操作指引_ReadyPlayerMe_Mixamo.md)

---

## 🔧 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 面部动画（视频驱动） | [LivePortrait](https://github.com/KwaiVGI/LivePortrait) | 快手可灵出品，SOTA 效果（独立工具） |
| 表情/头动生成 | [SadTalker](https://github.com/OpenTalker/SadTalker) | 音频驱动表情、眨眼与头部运动 |
| 面部动画（音频驱动） | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | 经典对口型模型，精修嘴型 |
| 人脸修复 | [GFPGAN](https://github.com/TencentARC/GFPGAN) | 消除 Wav2Lip 嘴部模糊，逐帧修复 + 检测框平滑 |
| 语音合成 | [Edge-TTS](https://github.com/rany2/edge-tts) | 微软 Edge 免费 TTS，国内可用 |
| 3D 渲染 | [Three.js](https://threejs.org/) | 浏览器端 WebGL 实时渲染 |
| 3D Avatar | [Ready Player Me](https://readyplayer.me) | AI 生成 3D 形象 |
| 3D 动作 | [Mixamo](https://www.mixamo.com) | Adobe 2 万+ 现成动作 |
| 视频合成 | FFmpeg | 音频/视频处理 |
| GUI | Gradio | Python 快速界面 |
| 后端 | FastAPI（规划）/ http.server（当前） | API 服务 |

---

## 📊 2D vs 3D 两套流水线对比

| 维度 | 2D 流水线 | 3D 流水线 |
|------|-----------|-----------|
| **技术栈** | Python + PyTorch | JavaScript + Three.js |
| **输入** | 单张照片 | Ready Player Me 3D Avatar |
| **面部** | SadTalker（表情/头动）+ Wav2Lip（嘴型）+ GFPGAN（修复） | Web Audio → jawOpen blendshape |
| **身体** | ❌ 不支持 | ✅ Mixamo 全身动作（跳舞等） |
| **输出** | 2D 视频 MP4 | 浏览器实时渲染 / 录屏 |
| **硬件要求** | 可纯 CPU | 浏览器 WebGL 即可 |
| **效果上限** | 好（2D SOTA） | 中（Web 渲染）→ UE5 升级后极好 |

---

## 🛣️ 升级方向（暂不实现，记录备忘）

| 目标 | 方案 | 硬件 | 复杂度 |
|------|------|------|--------|
| **实时数字人（直播/客服）** | UE5 + MetaHuman + NVIDIA Audio2Face + Live Link Face | RTX 4070+ | ⭐⭐⭐⭐⭐ |
| **商用级离线渲染** | Blender + Cycles + 云 GPU（AutoDL） | 云 GPU，几块钱/小时 | ⭐⭐⭐ |
| **多用户 API 服务** | FastAPI + 队列 + GPU Worker 池 | 多 GPU 服务器 | ⭐⭐⭐⭐ |
| **iOS/Android App** | Unity + IL2CPP + OVRLipSync | 移动设备 | ⭐⭐⭐ |

实时数字人目标架构：

```
iPhone Live Link Face (表情+头部)     NVIDIA Audio2Face (口型 viseme)
        └──────────────┬────────────────────┘
                       ▼
                UE5 Live Link 总控
                       ▼
                MetaHuman Avatar
                       ▼
                 ┌────┴────┐
                 ▼         ▼
             LLM 对话    NDI 直播输出
           (Ollama)    → OBS → 抖音/快手
```

---

## 📚 详细文档

- [项目信息.md](项目信息.md) — 模型原理、硬件清单、conda 环境、推理流水线详解
- [3d_avatar/README.md](3d_avatar/README.md) — 3D 子项目说明
- [3d_avatar/操作指引_ReadyPlayerMe_Mixamo.md](3d_avatar/操作指引_ReadyPlayerMe_Mixamo.md) — 3D 素材准备 step-by-step

---

## ⚠️ 注意事项

- 预训练模型文件较大（LivePortrait ~628MB、Wav2Lip ~872MB、SadTalker 256/512 各 ~692MB、GFPGAN ~520MB），首次运行前需下载
- 模型文件已通过 `.gitignore` 排除，新机器上首次运行时再下载
- 国内用户访问 HuggingFace 建议设置 `HF_ENDPOINT=https://hf-mirror.com`
- LivePortrait、Wav2Lip、SadTalker 各自保留原始 `.git` 目录便于单独追踪上游更新（源码同时纳入主仓库，模型除外）
- conda 环境的 ffmpeg 存在 DLL 冲突（`找不到 libintl_dgettext`），流水线自动优先使用 `ffmpeg/ffmpeg.exe` 本地静态版

## 📝 License

本项目整合多个开源组件，各组件遵循其原始许可：

- LivePortrait: [Apache-2.0](https://github.com/KwaiVGI/LivePortrait)
- Wav2Lip: [MIT](https://github.com/Rudrabha/Wav2Lip)
- Ready Player Me Avatar: 遵循 [RPM Terms](https://readyplayer.me/terms)
- Mixamo Animations: 遵循 [Adobe Terms](https://www.mixamo.com/#/?page=1&type=Motion%20Pack)

使用前请自行确认各组件的许可条款。
