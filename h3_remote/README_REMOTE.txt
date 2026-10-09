AutoDL 唱跳视频生成操作手册
============================

前置（本机网页操作）：
  1. AutoDL 注册充值（先充 10~20 元足够）: https://www.autodl.com
  2. 租一台 RTX 4090（24G），镜像选 PyTorch 2.x + CUDA 12.x 任意版本
  3. 开机后进入 JupyterLab，打开终端（Terminal）

远程执行（在实例终端里，逐条粘贴）：
  # 1. 上传 remote_bundle.zip 后（网页拖拽上传即可）：
  unzip remote_bundle.zip -d sdwork && cd sdwork
  # 2. 装环境 + 下模型（首次约 20~40 分钟，42GB 权重走平台内网很快）
  bash setup_h3.sh
  # 3. 逐段生成（每段几分钟，全程自动；失败段单独重跑：python run_segments.py --only 3）
  python run_segments.py
  # 4. 下载产物：JupyterLab 文件面板进入 out/，下载 segments.zip
  # 5. 建议手动关机停止计费（或在第 3 步前 export AUTODL_AUTO_SHUTDOWN=1 自动关机）

回到本机：
  把 segments.zip 解压到 output/sd_projects/<项目名>/segments/
  python song_dance_pipeline.py assemble --project <项目名>

常见问题：
  - setup_h3.sh 第 3 步下载失败：换镜像源 export HF_ENDPOINT=https://hf-mirror.com 后重跑，
    或用 export H3_MODEL_REPO=<实际仓名> 覆盖默认仓库
  - run_segments.py selftest 报节点不存在：说明工作流模板与已装 ComfyUI 的 H3 节点不匹配。
    打开 ComfyUI 网页(应用端口 8188)加载官方 MiniMax H3 模板 → 导出 API 格式 JSON
    替换 h3_workflow_api.json → 按实际节点号修改 h3_patch_map.json（run_segments.py 不用动）
  - 显存不足：确认租的是 24G 卡；H3 int8 在 12G 也能跑，可参考官方低显存方案 WanGP
