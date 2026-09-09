# 3D 数字人（第一阶段）

基于 Three.js 的浏览器端 3D 数字人，支持全身动作 + 音频驱动口型。

## 快速开始

```powershell
# 1. 启动后端（提供静态文件 + TTS API）
conda activate LivePortrait
cd d:\Work\Python\ai_digital_human\3d_avatar
python server.py

# 2. 浏览器打开
# http://localhost:8080/index.html
```

## 准备素材

### 1. 3D Avatar（必需）

- 打开 https://readyplayer.me
- 上传一张你的照片 → AI 生成 3D 形象
- 导出格式选 **glTF (.glb)**，**带 BlendShape**
- 把下载的 `.glb` 放到 `avatar/` 目录

### 2. Mixamo 动作（可选，想要跳舞/挥手等全身动作）

- 打开 https://www.mixamo.com
- 选择动作（Dancing / Talking / Walking / Waving 等）
- 下载格式选 **FBX for Unity** 或 **FBX Binary**
- 把下载的 `.fbx` 放到 `animations/` 目录

## 工作原理

```
┌─────────────────┐    ┌─────────────────────────────────┐
│  Python 后端     │    │  Three.js 前端（浏览器）            │
│  server.py       │    │  index.html                       │
│                 │    │                                   │
│  /api/files     │───▶│  加载 GLB Avatar + FBX 动作        │
│  /api/tts       │───▶│  Edge-TTS 生成音频 → Web Audio API │
│  audio/ 静态文件 │    │  音频能量 → jawOpen blendshape      │
└─────────────────┘    └─────────────────────────────────┘
```

## 口型驱动技术细节

Three.js 前端有两套口型应用机制：

1. **BlendShape（优先）**：Ready Player Me Avatar 内置 52 个 ARKit BlendShape
   - `jawOpen` index ≈ 33
   - 音频能量 → `jawOpen` influence
2. **骨骼旋转（fallback）**：如果 Avatar 没有 BlendShape
   - 找 jaw / head bone → rotation.x = -mouth * 0.5

## 支持的 Mixamo 动作

| 动作关键词 | 说明 |
|-----------|------|
| Dancing / Dance | 跳舞（复杂全身） |
| Talking | 说话（手势 + 头部） |
| Waving | 挥手 |
| Walking | 走路 |
| Idle | 待机 |
| Jumping | 跳跃 |

## 硬件建议

| 场景 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 开发调试（1 个 Avatar + 1 个动作） | CPU + 4GB 内存 | 任何 |
| 平滑播放（WebGL 实时） | 集显（UHD 730 可跑） | 独显 RTX 2060 |
| 复杂场景（多人 + 多动作） | 独显 RTX 3060 | RTX 4060 Ti |

**当前机器（Intel UHD 730）可以跑开发调试，但浏览器 WebGL 性能有限。**

## 升级方向（暂不实现，记录备忘）

| 目标 | 方案 | 硬件 | 复杂度 |
|------|------|------|--------|
| **实时数字人** | UE5 + MetaHuman + Audio2Face + Live Link Face | NVIDIA RTX 4070+ | ⭐⭐⭐⭐⭐ |
| **商用级渲染** | Blender + Cycles + 离线渲染 farm | 云 GPU（AutoDL） | ⭐⭐⭐ |
| **多用户/API 服务** | FastAPI + 队列 + GPU worker | 多 GPU 服务器 | ⭐⭐⭐⭐ |
| **iOS/Android App** | Unity + IL2CPP + OVRLipSync | 移动设备 | ⭐⭐⭐ |

## 文件清单

```
3d_avatar/
├── index.html          ← Three.js 前端（主要代码在这里）
├── server.py           ← Python 后端（Edge-TTS + 静态服务）
├── avatar/             ← 放 .glb Avatar 文件
├── animations/         ← 放 .fbx Mixamo 动作文件
└── audio/              ← TTS 生成的音频自动放这里
```
