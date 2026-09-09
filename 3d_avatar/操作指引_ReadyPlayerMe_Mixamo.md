# Ready Player Me + Mixamo 操作指引

> 目标：在 10 分钟内拿到一个能被 Three.js 直接加载的 `.glb` Avatar 和 `.fbx` 全身动作

---

## 目录

- [Part 1: Ready Player Me — 创建 3D Avatar](#part-1-ready-player-me--创建-3d-avatar)
- [Part 2: Mixamo — 下载全身动作](#part-2-mixamo--下载全身动作)
- [Part 3: 放入项目目录](#part-3-放入项目目录)
- [Part 4: 常见问题](#part-4-常见问题)

---

## Part 1: Ready Player Me — 创建 3D Avatar

### 前置条件

- 浏览器（Chrome/Edge 最新版）
- 一张清晰的人脸照片（可选，用于 AI 生成）
- 免费注册账号

### Step 1: 打开网站

浏览器访问 **https://readyplayer.me**

### Step 2: 选择创建方式

首页有两种创建方式：

| 方式 | 说明 | 耗时 | 相似度 |
|------|------|------|--------|
| **📸 Upload Photo** | 上传一张正面人脸照 → AI 自动生成 | ⭐ **最快 1 分钟** | 80-90% |
| **👤 Start from Scratch** | 手动选发型/脸型/衣服/配饰 | 5-10 分钟 | 可控 |

**推荐选 Upload Photo**，速度最快、相似度也够用。

### Step 3: 上传照片并调整

1. 点击 **Upload Photo** 按钮
2. 选择一张**正面、光照好、无遮挡**的人脸照片
3. 等待 30 秒，AI 自动生成 3D 形象
4. 预览生成效果，如有需要可以微调：
   - 左右方向可以切换发型、眼睛颜色、衣服等
   - 点 **Randomize** 可以随机生成风格
   - 点 **Next** 进入导出

### Step 4: 注册/登录

首次使用需要免费注册账号（支持 Google、邮箱登录）。

### Step 5: 下载 Avatar（关键！）

进入导出页面后，有**两种下载方式**，务必选对：

#### ⚠️ 方式 A: Download button（基础版）

直接点页面上的 **Download** 按钮 → 选 **glTF** → 得到一个 `.glb` 文件。

这个版本**不包含完整的面部 BlendShape**（只有基础形变），口型驱动效果会打折扣。

#### ✅ 方式 B: URL 带参数（完整版，推荐）

Ready Player Me 导出的 URL 支持 query 参数，可以指定带 **ARKit BlendShape** 的完整版：

1. 先点页面上的 Download，拿到一个类似这样的 URL：
   ```
   https://models.readyplayer.me/xxxxxxxx-xxxx-xxxx.glb
   ```
2. **直接在浏览器里修改 URL**，加上参数：
   ```
   ?morphTargets=ARKit,Oculus+Visemes,Default&textureAtlas=none&textureSizeLimit=1024&textureFormat=png&pose=T
   ```
   
   完整示例：
   ```
   https://models.readyplayer.me/xxxxxxxx-xxxx-xxxx.glb
   ?morphTargets=ARKit,Oculus+Visemes,Default
   &textureAtlas=none
   &textureSizeLimit=1024
   &textureFormat=png
   &pose=T
   ```
   
3. 回车下载，得到一个**带 52 个 ARKit BlendShape** 的完整 `.glb`

#### 参数说明

| 参数 | 值 | 作用 |
|------|-----|------|
| `morphTargets` | `ARKit,Oculus+Visemes,Default` | **关键！** 包含面部表情和口型 blendshape |
| `textureAtlas` | `none` | 不合并贴图（减少材质丢失问题） |
| `textureSizeLimit` | `1024` | 贴图尺寸限制（越大越精细，文件也越大） |
| `textureFormat` | `png` | 贴图格式 |
| `pose` | `T` | T-Pose（混合姿态，方便动作驱动） |

### 下载后验证

用任何 GLB 查看器（或 Blender）打开下载的文件，确认：

1. **文件大小**：完整版应该 > 5MB（基础版可能只有 2-3MB）
2. **有 MorphTarget**：在 Blender 里看 mesh 属性，应该能看到 52 个左右的 blendshape
3. **T-Pose**：手臂向两侧平举，手掌朝前（标准 T 姿态）

---

## Part 2: Mixamo — 下载全身动作

### 前置条件

- **Adobe ID**（免费注册）— Mixamo 是 Adobe 旗下服务
- 或者直接用 **Mixamo 上预置的角色**（不需要上传自己的 Avatar）

### 方案 A: 直接用 Mixamo 预置角色（最快）

如果你只是想先测试 Three.js 管线，不需要和 Ready Player Me 的真人形象绑定：

1. 打开 **https://www.mixamo.com**
2. 左侧选一个预置角色（比如 X Bot、Y Bot）
3. 顶部切到 **Animations** 标签
4. 搜索动作关键词，比如：
   - `dance` → 各种跳舞
   - `talking` → 说话带手势
   - `wave` → 挥手
   - `walk` / `run` → 走路/跑步
   - `idle` → 待机
5. 右侧可以调整参数：
   - **In Place**：勾选让角色在原地动（不位移），适合数字人
   - **Loop**：勾选让动作循环播放
   - **Overdrive**：调整动作幅度
   - **Arm Spacing**：调整手臂间距
6. 满意后点 **Download**，关键设置见下文

### 方案 B: 用自己的 Ready Player Me Avatar

> ⚠️ 重要：Mixamo 只接受 **FBX / OBJ** 格式上传，不接受 GLB！
> 所以 Ready Player Me 的 `.glb` 需要先转成 FBX，推荐用 Blender（免费，5 分钟）。

#### B-1: GLB → FBX（Blender 转换）

如果没装 Blender，去 https://www.blender.org 下载免费安装。

1. 打开 Blender，删除默认的立方体
2. **File → Import → glTF 2.0 (.glb)**，选择你的 Ready Player Me Avatar
3. 导入后确认：角色 T-Pose，脚底在 y=0 附近
4. **File → Export → FBX**，导出设置：
   - 格式：**FBX Binary (.fbx)**
   - Scale：**1.0**
   - Forward：**-Z Forward**
   - Up：**Y Up**
   - 勾选 **Apply Transform**
   - 勾选 **Only Deform Bones**（只导出变形骨骼）
5. 保存，得到一个 `.fbx` 文件

#### B-2: 上传到 Mixamo

1. 打开 **https://www.mixamo.com**，登录 Adobe ID
2. 点右上角 **Upload Character** 按钮
3. 上传刚才导出的 `.fbx`
4. Mixamo 会自动识别骨骼（如果报错，需要手动点 3 个关键关节位置）
5. 等待 10-30 秒处理，你的 Avatar 就出现在预览里了

#### B-3: 给 Avatar 选动作

1. 顶部切到 **Animations** 标签
2. 搜索动作（推荐关键词见上）
3. 点一个动作 → 右侧预览 → 调整参数（In Place、Loop 等）
4. 满意后点 **Download**

### Download 对话框（关键设置）

下载窗口的设置直接影响 Three.js 能不能正确加载：

```
╔══════════════════════════════════════════╗
║  Download 对话框                          ║
║                                          ║
║  Format:      FBX Binary (.fbx)    ✅ 选这个 ║
║               FBX ASCII (.fbx)            ║
║               COLLADA (.dae)              ║
║                                          ║
║  Skin:        With Skin           ✅ 选这个 ║
║               Without Skin                ║
║                                          ║
║  Frames Per Second: 30            ✅       ║
║                     24 / 60               ║
║                                          ║
║  Keyframe Reduction: None         ✅ 选这个 ║
║                      Low / Medium / High  ║
║                                          ║
║                    [ Download ]            ║
╚══════════════════════════════════════════╝
```

#### 为什么这样选？

| 设置 | 选什么 | 原因 |
|------|--------|------|
| **Format** | FBX Binary | Three.js FBXLoader 原生支持；体积小；Blender/Unity/UE 通用 |
| **Skin** | With Skin | ✅ **包含 Mesh + 骨骼 + 动画全部**。Without Skin 只有骨骼动画没有模型，会导致 Three.js 加载时材质丢失 |
| **FPS** | 30 | 标准帧率，和 Three.js 默认一致 |
| **Keyframe Reduction** | None | 不压缩关键帧，保持动画流畅度 |

### 动作推荐清单

| 搜索关键词 | 得到的动作 | 适用场景 |
|-----------|-----------|---------|
| `dance` | Hip Hop、Breakdance、Ballet、TikTok | 复杂全身动作展示 |
| `talking` | 说话带手势、演讲、解说 | 数字人"说话"场景 |
| `wave` | 挥手、打招呼 | 开场/欢迎 |
| `walk` / `run` | 走路、跑步 | 走位场景 |
| `idle` | 待机、呼吸 | 背景状态 |
| `jump` | 跳跃、翻跟头 | 动作展示 |
| `celebrate` | 庆祝、欢呼 | 情绪表达 |

---

## Part 3: 放入项目目录

### 放好后的目录结构

```
3d_avatar/
├── avatar/                          ← Ready Player Me 导出
│   └── my_avatar.glb                ✅ 放这里（文件名随便起）
│
├── animations/                      ← Mixamo 下载
│   ├── hip_hop_dance.fbx            ✅ 放这里
│   ├── talking_gesture.fbx          ✅ 放这里
│   └── wave.fbx                     ✅ 放这里
│
├── audio/                           ← 自动生成
├── index.html                       Three.js 前端
├── server.py                        Python 后端
└── README.md
```

### 启动并验证

```powershell
conda activate LivePortrait
cd d:\Work\Python\ai_digital_human\3d_avatar
python server.py
# 浏览器打开 http://localhost:8080/index.html
```

页面加载后：
1. **Avatar 下拉框** 应该自动出现 `my_avatar.glb`
2. **动作下拉框** 应该自动出现所有 `.fbx`
3. 选好 Avatar 和动作 → 点"🚀 生成语音 + 播放" → 数字人就会说话 + 做全身动作

---

## Part 4: 常见问题

### Q1: Ready Player Me 下载的 GLB 只有 2KB？

**原因**：可能没带 BlendShape 参数。

**解决**：用方式 B，在 URL 后面加 `?morphTargets=ARKit,Oculus+Visemes,Default&textureAtlas=none&textureSizeLimit=1024&textureFormat=png&pose=T`，正常大小应该 > 5MB。

### Q2: Mixamo 上传 Ready Player Me Avatar 报 "Unable to map skeleton"？

**原因**：GLB 格式 Mixamo 不识别骨骼。

**解决**：用 Blender 把 GLB 转成 FBX Binary 再上传。见 Part 2 B-1 节。

### Q3: Mixamo 下载的 FBX 用 Three.js 加载后角色是灰色的？

**原因**：下载时选了 `Without Skin`，FBX 里没有 mesh 和材质。

**解决**：重新下载，**Skin 选 With Skin**。

### Q4: 动作加载了但角色没动？

**原因 1**：`AnimationMixer` 没有同时管理 Avatar 和 FBX 的动画。

**解决**：Three.js 代码里要确保 FBX 的 animation clip 通过同一个 `AnimationMixer` 应用到 Avatar 上（当前代码已处理）。

**原因 2**：FBX 骨骼名称和 GLB Avatar 的骨骼名称不匹配。

**解决**：这种情况很常见，因为不同来源的 3D 资产骨骼命名不一样（比如一个叫 `mixamorig:Hips` 另一个叫 `Hips`）。当前代码还没做自动骨骼重定向，这是后续可以优化的方向。

### Q5: 3D 角色加载后面朝后？

**原因**：GLB 的 forward 轴和 Three.js 默认不一致。

**解决**：在 index.html 的 `loadAvatar` 函数里，加载完后对 avatar 做一次 `avatar.rotation.y = Math.PI`（旋转 180 度）。

### Q6: 想让动作只动身体不动脸？

Mixamo 的动作主要是身体骨骼，面部表情很少，不需要特别处理。身体动作和面部口型是独立的两套动画层，Three.js 会自动叠加。

### Q7: 一个 Avatar 能套多个动作吗？

可以。Mixamo 下载多个 `.fbx`，都放到 `animations/` 目录，Three.js 会同时加载，支持切换。

### Q8: 怎么录屏生成最终的 MP4？

浏览器里的数字人是实时渲染的，要录屏有两个方案：

1. **OBS Studio**（免费）：窗口捕获 → 选浏览器窗口 → 录制
2. **FFmpeg 录屏**（需要额外配置设备）

---

## 快速核对清单

| 步骤 | 完成标志 | 状态 |
|------|---------|------|
| Ready Player Me 注册 | 有账号 | ☐ |
| 创建 Avatar | 拿到 `.glb`（>5MB） | ☐ |
| URL 带 BlendShape 参数 | GLB 包含 ARkit 52 shape | ☐ |
| Mixamo 注册 | 有 Adobe ID | ☐ |
| 选 3-5 个动作下载 | 拿到 `.fbx`（FBX Binary, With Skin） | ☐ |
| 放到项目目录 | `avatar/*.glb` + `animations/*.fbx` | ☐ |
| 启动 server.py | http://localhost:8080 可访问 | ☐ |
| Three.js 页面加载成功 | Avatar 和动作都出现在下拉框 | ☐ |
| 口型同步工作 | 点"生成语音 + 播放"能看到嘴动 | ☐ |

---

**参考链接**

- Ready Player Me: https://readyplayer.me
- Mixamo: https://www.mixamo.com
- Blender（GLB → FBX 转换）: https://www.blender.org
- Three.js FBX 文档: https://threejs.org/docs/#examples/en/loaders/FBXLoader
- ARKit 52 BlendShape 说明: https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
