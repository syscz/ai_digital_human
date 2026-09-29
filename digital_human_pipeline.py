# coding: utf-8
"""
AI 数字人端到端流水线
=====================
输入：源图片 + 文本
输出：说话视频（嘴型跟音频同步）

流水线：文本 → Edge-TTS(语音合成) → SadTalker(表情/头动) → Wav2Lip(精修嘴型) → GFPGAN(人脸修复) → MP4

使用方式：
  # 命令行
  python digital_human_pipeline.py --photo photo.jpg --text "你好，我是数字人" --voice zh-CN-XiaoxiaoNeural --output ./output

  # Gradio 界面
  python digital_human_pipeline.py --gradio
"""

import argparse
import asyncio
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).parent.resolve()
WAV2LIP_DIR = PROJECT_ROOT / "Wav2Lip"
WAV2LIP_INFERENCE = WAV2LIP_DIR / "inference.py"
# GAN 版模型嘴部更清晰、口型更稳定；若缺失则回退到普通版
WAV2LIP_CHECKPOINT_GAN = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
WAV2LIP_CHECKPOINT = (
    WAV2LIP_CHECKPOINT_GAN
    if WAV2LIP_CHECKPOINT_GAN.exists()
    else WAV2LIP_DIR / "checkpoints" / "wav2lip.pth"
)
WAV2LIP_TEMP = WAV2LIP_DIR / "temp"
SADTALKER_DIR = PROJECT_ROOT / "SadTalker"
SADTALKER_INFERENCE = SADTALKER_DIR / "inference.py"
GFPGAN_WEIGHTS = PROJECT_ROOT / "gfpgan_weights" / "GFPGANv1.4.pth"
CONDA_ENV = "LivePortrait"

# 常用中文语音
CHINESE_VOICES = {
    "女声-亲切": "zh-CN-XiaoxiaoNeural",
    "女声-温柔": "zh-CN-XiaoyiNeural",
    "男声-新闻": "zh-CN-YunxiNeural",
    "男声-年轻": "zh-CN-YunjianNeural",
    "男声-稳重": "zh-CN-YunyangNeural",
}


def get_ffmpeg_exe() -> str:
    """
    返回可用的 ffmpeg 可执行文件路径（绝对路径优先）。

    优先返回项目本地静态版 ffmpeg/ffmpeg.exe（imageio-ffmpeg 自带，
    不依赖 conda 的 DLL），不存在时回退到 PATH 里的 ``ffmpeg``。
    """
    local_ffmpeg = PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    return "ffmpeg"


def ensure_ffmpeg_in_path() -> bool:
    """
    把本地静态 ffmpeg 目录 prepend 到 PATH，使 Wav2Lip 等子进程里
    用 ``ffmpeg`` 字符串也能命中可用版本。

    背景：conda 环境的 ffmpeg 因 conda-forge(ffmpeg/fontconfig) 与
    defaults(glib/libglib) 混装，DLL 冲突，一执行就报
    ``找不到 libintl_dgettext``（退出码 0xC0000139）。
    项目本地 ffmpeg/ffmpeg.exe 是 imageio-ffmpeg 自带的静态编译版，
    不依赖这些 DLL，因此优先使用它。
    """
    local_ffmpeg = PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        bin_dir = str(local_ffmpeg.parent)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        return True
    return False


def check_prerequisites():
    """检查所有前置条件是否满足"""
    issues = []

    # 检查 Wav2Lip 源码
    if not WAV2LIP_INFERENCE.exists():
        issues.append(f"❌ Wav2Lip inference.py 不存在：{WAV2LIP_INFERENCE}")
    if not WAV2LIP_CHECKPOINT.exists():
        issues.append(f"❌ Wav2Lip 模型不存在：{WAV2LIP_CHECKPOINT}")

    # 检查 conda 环境
    result = subprocess.run(
        ["conda", "run", "-n", CONDA_ENV, "python", "-c", "import torch, edge_tts; print('OK')"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        issues.append(f"❌ conda 环境 '{CONDA_ENV}' 检查失败：{result.stderr.strip()[:200]}")

    # 检查 ffmpeg：优先使用本地静态版（绕开 conda 的 DLL 冲突）
    ffmpeg_ok = False
    if ensure_ffmpeg_in_path():
        # 用本地静态 ffmpeg 再验一次，确认真正可用
        local_ffmpeg = PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe"
        r = subprocess.run([str(local_ffmpeg), "-version"], capture_output=True, timeout=5)
        if r.returncode == 0:
            ffmpeg_ok = True
    if not ffmpeg_ok:
        # 回退到 PATH 里的 ffmpeg
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if r.returncode == 0:
            ffmpeg_ok = True
    if not ffmpeg_ok:
        issues.append("❌ FFmpeg 不可用：conda 环境 ffmpeg 存在 DLL 冲突，且本地静态 ffmpeg 缺失")

    if issues:
        for i in issues:
            print(i)
        return False

    print("✅ 前置条件检查通过")
    return True


# ========== Step 1: Edge-TTS 语音合成 ==========
async def text_to_speech(text: str, voice: str, output_wav: str) -> str:
    """
    用 Edge-TTS 把文本合成为语音（WAV 格式）

    Args:
        text: 要合成的文本
        voice: 语音名称（如 zh-CN-XiaoxiaoNeural）
        output_wav: 输出 wav 文件路径

    Returns:
        生成的 wav 文件路径
    """
    import edge_tts

    print(f"🎙️  Edge-TTS 合成语音...")
    print(f"   文本: '{text}'")
    print(f"   语音: {voice}")

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_wav)

    # Edge-TTS 默认输出 webm/mp3，需要转 wav（16kHz mono，Wav2Lip 需要）
    wav_16k = output_wav.replace(".wav", "_16k.wav")
    ffmpeg_cmd = [
        get_ffmpeg_exe(), "-y",
        "-i", output_wav,
        "-ar", "16000",  # 采样率 16kHz
        "-ac", "1",      # 单声道
        "-c:a", "pcm_s16le",  # 16-bit PCM
        wav_16k,
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    # 替换原文件
    shutil.move(wav_16k, output_wav)

    file_size = os.path.getsize(output_wav) / 1024
    print(f"   ✅ 语音生成: {output_wav} ({file_size:.0f} KB)")
    return output_wav


# ========== Step 2: Wav2Lip 音频驱动 ==========
def audio_to_video(
    face_input: str,
    audio_path: str,
    output_path: str,
    wav2lip_checkpoint: str = None,
    batch_size: int = 8,
    static_input: bool = True,
) -> str:
    """
    用 Wav2Lip 把照片/视频 + 音频合成为说话视频

    Args:
        face_input: 源图片或源视频路径（SadTalker 串联时传视频）
        audio_path: 音频文件路径（wav）
        output_path: 输出视频路径（mp4）
        wav2lip_checkpoint: Wav2Lip 模型路径
        batch_size: 推理 batch size（CPU 建议 8，GPU 可 128）
        static_input: 输入是否为静态图片（True 只用第一帧并关闭平滑；
                      False 时对视频做人脸框时序平滑，跟随头部运动）

    Returns:
        生成的视频文件路径
    """
    if wav2lip_checkpoint is None:
        wav2lip_checkpoint = str(WAV2LIP_CHECKPOINT)

    # 确保 Wav2Lip/temp/ 存在（inference.py 里硬编码用这个目录）
    WAV2LIP_TEMP.mkdir(parents=True, exist_ok=True)

    print(f"🎬  Wav2Lip 音频驱动...")
    print(f"   输入: {face_input}")
    print(f"   音频: {audio_path}")
    print(f"   模型: {wav2lip_checkpoint}")
    print(f"   Batch size: {batch_size}")

    cmd = [
        sys.executable, str(WAV2LIP_INFERENCE),
        "--checkpoint_path", wav2lip_checkpoint,
        "--face", face_input,
        "--audio", audio_path,
        "--outfile", output_path,
        "--pads", "0", "15", "0", "0",  # 底部多 padding 防止切到下巴
        "--wav2lip_batch_size", str(batch_size),
    ]
    if static_input:
        # 源是图片：只用第一帧，且无需时间平滑
        cmd += ["--static", "True", "--nosmooth"]

    print(f"   执行: {' '.join(cmd)}")

    # CWD 必须是 Wav2Lip 目录（因为 inference.py 用 import audio 等相对导入）
    result = subprocess.run(
        cmd,
        cwd=str(WAV2LIP_DIR),
        capture_output=False,  # 直接把输出打到控制台
    )

    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip 推理失败，退出码: {result.returncode}")

    if not os.path.exists(output_path):
        raise RuntimeError(f"Wav2Lip 未生成输出文件: {output_path}")

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"   ✅ 视频生成: {output_path} ({file_size:.1f} MB)")
    return output_path


# ========== Step 2.5 (可选): SadTalker 表情/头动生成 ==========
def generate_expressive_video(
    photo_path: str,
    audio_path: str,
    output_path: str,
    still: bool = False,
    expression_scale: float = 1.0,
    render_size: int = 512,
) -> str:
    """
    用 SadTalker 从音频生成带头部运动 + 眨眼 + 表情的说话视频。

    SadTalker 嘴型不够准，因此它的输出会再交给 Wav2Lip 修正嘴部，
    本步骤只负责"让脸动起来"。

    Args:
        photo_path: 源图片（支持半身照，full 模式保留原图构图）
        audio_path: 驱动音频（wav）
        output_path: 输出视频路径（mp4）
        still: True 时大幅减少头部摆动（更稳但略生硬）
        expression_scale: 表情幅度，1.0 为默认
        render_size: 内部渲染分辨率 256 或 512；512 更贴近本人相貌、
            面部细节更好，但 CPU 渲染耗时约为 256 的 3-4 倍
            （需要 checkpoints/SadTalker_V0.0.2_512.safetensors）

    Returns:
        生成的视频文件路径
    """
    if not SADTALKER_INFERENCE.exists():
        raise RuntimeError(f"未找到 SadTalker：{SADTALKER_INFERENCE}")

    model_file = SADTALKER_DIR / "checkpoints" / f"SadTalker_V0.0.2_{render_size}.safetensors"
    if not model_file.exists():
        raise RuntimeError(
            f"缺少 SadTalker {render_size} 模型：{model_file}。"
            f"可从 ModelScope wwd123/sadtalker 下载 checkpoints/SadTalker_V0.0.2_{render_size}.safetensors"
        )

    # SadTalker 输出文件名不可控，用独立临时目录接收，再搬走唯一 mp4
    result_dir = tempfile.mkdtemp(prefix="sadtalker_")

    print("🎭  SadTalker 生成表情与头部运动（CPU 较慢，请耐心等待）...")
    print(f"   照片: {photo_path}")
    print(f"   音频: {audio_path}")
    print(f"   渲染分辨率: {render_size}")

    cmd = [
        sys.executable, str(SADTALKER_INFERENCE),
        "--driven_audio", audio_path,
        "--source_image", photo_path,
        "--result_dir", result_dir,
        "--preprocess", "full",   # 保留原图构图（身体/背景），适合半身照
        "--size", str(render_size),
        "--expression_scale", str(expression_scale),
        "--cpu",
    ]
    if still:
        cmd.append("--still")

    # SadTalker 结尾用裸 `ffmpeg` 命令（os.system）合并音轨，
    # 必须保证 PATH 里是项目本地静态 ffmpeg，而不是 conda 的 DLL 冲突版
    ensure_ffmpeg_in_path()

    # CWD 必须是 SadTalker 目录（相对路径的 checkpoints/、gfpgan/weights、src 包）
    result = subprocess.run(cmd, cwd=str(SADTALKER_DIR), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"SadTalker 生成失败，退出码: {result.returncode}")

    mp4s = list(Path(result_dir).rglob("*.mp4"))
    if not mp4s:
        raise RuntimeError(f"SadTalker 未生成视频，结果目录: {result_dir}")

    shutil.move(str(mp4s[0]), output_path)
    shutil.rmtree(result_dir, ignore_errors=True)

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"   ✅ 表情视频生成: {output_path} ({file_size:.1f} MB)")
    return output_path


# ========== Step 3 (可选): GFPGAN 人脸修复 ==========
_GFPGAN_RESTORER = None  # 模块级缓存，Gradio 多次生成不重复加载模型
# mode:
#   "fixed"  = 固定框，适用于 Wav2Lip --static（人脸位置全程不变），只检测第一帧
#   "smooth" = 逐帧检测 + EMA 指数平滑，适用于 SadTalker 头动视频；
#              既跟得上头部运动，又避免逐帧检测框抖动；某帧检测失败则沿用上一帧
_FACE_DET_CACHE = {"mode": "fixed", "bboxes": None, "prev": None, "alpha": 0.6}


def _get_gfpgan_restorer():
    """懒加载 GFPGANer（只修复不放大，背景不做超分），进程内只加载一次。

    retinaface 人脸检测做了 monkey-patch，行为由 _FACE_DET_CACHE["mode"] 控制：
    - fixed：静态图链路复用第一帧检测框（省掉逐帧检测耗时）；
    - smooth：头动视频每帧检测，对检测框做 EMA 平滑（检测失败沿用上一帧）。
    """
    global _GFPGAN_RESTORER
    if _GFPGAN_RESTORER is None:
        import numpy as np
        from gfpgan import GFPGANer

        _GFPGAN_RESTORER = GFPGANer(
            model_path=str(GFPGAN_WEIGHTS),
            upscale=1,            # 不放大分辨率，只做细节修复
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,    # 背景超分很慢且非必需，关闭
        )

        face_det = _GFPGAN_RESTORER.face_helper.face_det
        orig_detect = face_det.detect_faces

        def _patched_detect(img, threshold, _orig=orig_detect, _cache=_FACE_DET_CACHE):
            if _cache["mode"] == "fixed":
                # 静态图：人脸不动，第一帧检测一次后全程复用
                if _cache["bboxes"] is None:
                    _cache["bboxes"] = _orig(img, threshold)
                return _cache["bboxes"]

            # 头动视频：逐帧检测，取置信度最高的人脸做 EMA 平滑
            raw = _orig(img, threshold)
            cur = None
            if raw is not None and len(raw) > 0:
                cur = np.asarray(raw[0], dtype=np.float64)  # [x1,y1,x2,y2,score]

            prev = _cache["prev"]
            if cur is not None:
                if prev is None:
                    sm = cur.copy()
                else:
                    alpha = _cache["alpha"]
                    sm = prev.copy()
                    sm[:4] = (1.0 - alpha) * prev[:4] + alpha * cur[:4]
                    sm[4] = cur[4]  # 置信度列不平滑
                _cache["prev"] = sm
            else:
                # 侧脸/眨眼等导致本帧检测失败：沿用上一帧平滑框，避免框跳丢
                sm = prev

            return None if sm is None else sm[None, :]

        face_det.detect_faces = _patched_detect
    return _GFPGAN_RESTORER


def enhance_video(
    video_path: str,
    fixed_face_box: bool = False,
    gfpgan_weight: float = 0.5,
    interpolate_fps: int = None,
) -> str:
    """
    用 GFPGAN 对视频逐帧做人脸修复（原地替换），解决 Wav2Lip
    嘴部 96×96 上采样带来的模糊。

    逐帧流式处理（读一帧修一帧写一帧），避免整段视频驻留内存。

    Args:
        fixed_face_box: True 时人脸框只检测第一帧并全程复用
            （仅适用于人脸不动的静态图链路）；False 时逐帧检测并做 EMA
            平滑，适用于 SadTalker 头动视频——否则错位的固定框会把
            每帧的脸"扶正"成不同样子，既改相貌又造成帧间跳动。
        gfpgan_weight: 修复力度 0~1，越小越贴近原图（保留本人相貌），
            越大去模糊越强但越可能"换脸"，默认 0.5。
        interpolate_fps: 设置后（如 50）用 ffmpeg minterpolate 运动补偿
            插帧到该帧率，让头动/说话更顺滑；失败时自动降级为不插帧。
    """
    if not GFPGAN_WEIGHTS.exists():
        print("   ⚠️  未找到 GFPGAN 模型（gfpgan_weights/GFPGANv1.4.pth），跳过人脸修复")
        return video_path

    import cv2

    mode_desc = "固定框（静态图）" if fixed_face_box else "逐帧检测+EMA平滑（头动）"
    print(f"✨  GFPGAN 人脸修复（力度 {gfpgan_weight}，{mode_desc}，CPU 上每帧约数秒）...")

    # facexlib 的辅助模型路径是相对 CWD 的 gfpgan/weights，切到项目根保证能找到
    old_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        restorer = _get_gfpgan_restorer()
    finally:
        os.chdir(old_cwd)

    # 每个新视频重置检测状态，并切换框模式
    _FACE_DET_CACHE["mode"] = "fixed" if fixed_face_box else "smooth"
    _FACE_DET_CACHE["bboxes"] = None
    _FACE_DET_CACHE["prev"] = None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = video_path.replace(".mp4", "_gfpgan_tmp.mp4")
    final_tmp = video_path.replace(".mp4", "_final_tmp.mp4")
    writer = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, _, restored = restorer.enhance(
                frame, has_aligned=False, only_center_face=False,
                paste_back=True, weight=gfpgan_weight,
            )
            writer.write(restored)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"   已修复 {frame_idx} 帧...")
    finally:
        cap.release()
        writer.release()

    # ffmpeg 压缩修复后的帧（mp4v 压缩率差）并从原视频拷贝音频
    base_ffmpeg_cmd = [
        get_ffmpeg_exe(), "-y",
        "-i", tmp_path,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:a", "copy",
        "-shortest",
    ]

    if interpolate_fps:
        # 运动补偿插帧（双向运动估计+重叠块补偿），让帧间运动更顺滑
        vf = (
            f"minterpolate=fps={interpolate_fps}:mi_mode=mci:"
            f"mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        )
        print(f"🎞️  运动补偿插帧 → {interpolate_fps}fps（CPU 较慢）...")
        result = subprocess.run(
            [
                *base_ffmpeg_cmd[:8],
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                *base_ffmpeg_cmd[8:],
                final_tmp,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            # 插帧失败不阻断主流程，降级为直接压缩
            print("   ⚠️  插帧失败，降级为不插帧输出")
            final_tmp = video_path.replace(".mp4", "_nointerp_tmp.mp4")
            subprocess.run(
                [
                    *base_ffmpeg_cmd[:8],
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    *base_ffmpeg_cmd[8:],
                    final_tmp,
                ],
                capture_output=True, check=True,
            )
    else:
        subprocess.run(
            [
                *base_ffmpeg_cmd[:8],
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                *base_ffmpeg_cmd[8:],
                final_tmp,
            ],
            capture_output=True, check=True,
        )

    os.replace(final_tmp, video_path)  # 原地替换
    os.remove(tmp_path)

    file_size = os.path.getsize(video_path) / 1024 / 1024
    print(f"   ✅ 人脸修复完成: {frame_idx} 帧 ({file_size:.1f} MB)")
    return video_path


# ========== 主流水线 ==========
async def run_pipeline(
    photo_path: str,
    text: str,
    voice: str = "zh-CN-XiaoxiaoNeural",
    output_dir: str = None,
    wav2lip_checkpoint: str = None,
    enable_sadtalker: bool = True,
    sadtalker_still: bool = False,
    sadtalker_size: int = 512,
    enable_gfpgan: bool = True,
    gfpgan_weight: float = 0.5,
    enable_interpolation: bool = True,
    keep_intermediate: bool = False,
) -> dict:
    """
    端到端流水线：文本 + 照片 → 说话视频

    完整链路：
        Edge-TTS → SadTalker(表情/头动) → Wav2Lip(精修嘴型) → GFPGAN(人脸修复) → 插帧
    关闭 SadTalker 时退化为：Edge-TTS → Wav2Lip（仅嘴动）

    Args:
        photo_path: 源图片路径
        text: 要说的话
        voice: Edge-TTS 语音名称
        output_dir: 输出目录（默认在 output/ 下）
        wav2lip_checkpoint: Wav2Lip 模型路径（默认优先 GAN 版）
        enable_sadtalker: 是否启用 SadTalker 表情/头动（让视频自然）
        sadtalker_still: SadTalker 是否减少头部摆动
        sadtalker_size: SadTalker 渲染分辨率 256/512（512 更像本人但更慢）
        enable_gfpgan: 是否用 GFPGAN 做人脸修复（解决嘴部模糊）
        gfpgan_weight: GFPGAN 修复力度 0~1（越小越保留本人相貌）
        enable_interpolation: 是否在头动链路末尾插帧到 50fps（更顺滑）
        keep_intermediate: 保留 SadTalker 中间视频（expressive_*.mp4）；
            默认流水线成功后删除，失败时始终保留以便排查

    Returns:
        dict: {"audio": "xxx.wav", "video": "xxx.mp4"}
    """
    # 准备输出目录
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确保 ffmpeg 优先用本地静态版（Gradio 模式不经过 check_prerequisites）
    ensure_ffmpeg_in_path()

    # 用时间戳避免文件名冲突
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_path = str(output_dir / f"speech_{timestamp}.wav")
    video_path = str(output_dir / f"talking_{timestamp}.mp4")

    print("\n" + "=" * 60)
    print(f"🚀 AI 数字人流水线启动")
    print(f"   照片: {photo_path}")
    print(f"   文本: '{text}'")
    print(f"   语音: {voice}")
    print(f"   输出: {output_dir}")
    print("=" * 60 + "\n")

    try:
        # Step 1: 语音合成
        await text_to_speech(text, voice, audio_path)

        sad_path = None
        if enable_sadtalker:
            # Step 2a: SadTalker 生成带表情/头动的视频（嘴型不准）
            sad_path = str(output_dir / f"expressive_{timestamp}.mp4")
            generate_expressive_video(
                photo_path, audio_path, sad_path,
                still=sadtalker_still, render_size=sadtalker_size,
            )
            # Step 2b: Wav2Lip 以表情视频为底，精修嘴型（视频输入，跟随头动）
            audio_to_video(
                sad_path, audio_path, video_path, wav2lip_checkpoint,
                static_input=False,
            )
        else:
            # 退化链路：照片直接驱动，只有嘴动
            audio_to_video(
                photo_path, audio_path, video_path, wav2lip_checkpoint,
                static_input=True,
            )

        # Step 3: 人脸修复（可选，改善嘴部模糊）
        if enable_gfpgan:
            # 头动视频必须逐帧检测+平滑框；静态图链路才用固定框
            # 插帧只在头动链路开启（静态嘴动没有帧间运动可补）
            enhance_video(
                video_path,
                fixed_face_box=not enable_sadtalker,
                gfpgan_weight=gfpgan_weight,
                interpolate_fps=50 if (enable_sadtalker and enable_interpolation) else None,
            )

        print("\n" + "=" * 60)
        print(f"🎉 流水线完成！")
        print(f"   音频: {audio_path}")
        print(f"   视频: {video_path}")
        print("=" * 60 + "\n")

        # 成功后才清理中间视频；失败路径走 except，不删除（保留排查现场）
        if sad_path and not keep_intermediate:
            try:
                os.remove(sad_path)
                print(f"🧹  已清理中间文件: {sad_path}")
            except OSError:
                pass

        return {"audio": audio_path, "video": video_path}

    except Exception as e:
        print(f"\n❌ 流水线出错: {e}")
        traceback.print_exc()
        raise


# ========== CLI 入口 ==========
def main_cli():
    parser = argparse.ArgumentParser(description="AI 数字人端到端流水线")
    parser.add_argument("--photo", default=None, help="源图片路径 (jpg/png)")
    parser.add_argument("--text", default=None, help="要说的话")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural",
                        help="Edge-TTS 语音，默认 zh-CN-XiaoxiaoNeural")
    parser.add_argument("--output", default=None, help="输出目录 (默认 output/，文件名为 talking_时间戳.mp4)")
    parser.add_argument("--checkpoint", default=None, help="Wav2Lip 模型路径 (默认优先 checkpoints/wav2lip_gan.pth)")
    parser.add_argument("--no-gfpgan", action="store_true",
                        help="关闭 GFPGAN 人脸修复（默认开启，用于改善嘴部模糊）")
    parser.add_argument("--no-sadtalker", action="store_true",
                        help="关闭 SadTalker 表情/头动（默认开启；关闭后只有嘴动）")
    parser.add_argument("--sadtalker-still", action="store_true",
                        help="SadTalker 减少头部摆动（更稳但自然度略降）")
    parser.add_argument("--sadtalker-size", type=int, default=512, choices=[256, 512],
                        help="SadTalker 渲染分辨率（默认 512，更像本人；256 更快）")
    parser.add_argument("--no-interpolation", action="store_true",
                        help="关闭 50fps 运动插帧（默认开启，仅在 SadTalker 链路生效）")
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="保留 SadTalker 中间视频 expressive_*.mp4（默认成功后删除）")
    parser.add_argument("--gfpgan-weight", type=float, default=0.5,
                        help="GFPGAN 修复力度 0~1（默认 0.5，越小越保留本人相貌）")
    parser.add_argument("--gradio", action="store_true", help="启动 Gradio 界面")

    args = parser.parse_args()

    if args.gradio:
        launch_gradio()
        return

    # 命令行模式下 photo 和 text 必填
    if not args.photo or not args.text:
        parser.error("--photo 和 --text 是命令行模式下的必填参数（Gradio 模式用 --gradio 启动）")

    if not check_prerequisites():
        sys.exit(1)

    result = asyncio.run(run_pipeline(
        photo_path=args.photo,
        text=args.text,
        voice=args.voice,
        output_dir=args.output or str(PROJECT_ROOT / "output"),
        wav2lip_checkpoint=args.checkpoint,
        enable_sadtalker=not args.no_sadtalker,
        sadtalker_still=args.sadtalker_still,
        sadtalker_size=args.sadtalker_size,
        enable_gfpgan=not args.no_gfpgan,
        gfpgan_weight=args.gfpgan_weight,
        enable_interpolation=not args.no_interpolation,
        keep_intermediate=args.keep_intermediate,
    ))

    print(f"\n🎬 生成的视频: {result['video']}")


# ========== Gradio 入口 ==========
def launch_gradio():
    try:
        import gradio as gr
    except ImportError:
        print("请先安装 gradio: conda activate LivePortrait && pip install gradio")
        sys.exit(1)

    with gr.Blocks(title="AI 数字人 - 文本生成说话视频") as demo:
        gr.Markdown("# 🎬 AI 数字人\n上传一张照片 + 输入文字，生成说话视频（Wav2Lip + Edge-TTS）")

        with gr.Row():
            with gr.Column():
                photo = gr.Image(label="📷 源图片", type="pil")
                text = gr.Textbox(label="📝 要说的话", value="你好，我是数字人，很高兴见到你！", lines=3)
                voice = gr.Dropdown(
                    label="🎙️ 语音选择",
                    choices=list(CHINESE_VOICES.keys()),
                    value="女声-亲切",
                )
                enable_sadtalker = gr.Checkbox(
                    value=True,
                    label="🎭 自然表情与头部运动（SadTalker）",
                    info="开启：有眨眼、表情和头动，再由 Wav2Lip 精修嘴型，更自然但较慢；关闭：只有嘴动，速度快",
                )
                sadtalker_still = gr.Checkbox(
                    value=False,
                    label="减少头部摆动（更稳重的口播效果）",
                    interactive=True,
                )
                enable_interpolation = gr.Checkbox(
                    value=True,
                    label="🎞️ 50fps 运动插帧（动作更顺滑）",
                    info="用运动补偿把 25fps 补到 50fps，减少一帧一帧的卡顿感；仅在开启 SadTalker 时生效",
                )
                btn = gr.Button("🚀 生成视频", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="🎥 生成的说话视频")

        def process(image, text, voice_label, enable_sadtalker, sadtalker_still,
                    enable_interpolation):
            """Gradio 处理函数"""
            # Gradio 4.x 里 Image(type="pil") 返回 PIL.Image，需要转成临时文件
            photo_path = image if isinstance(image, str) else None
            tmp_name = None
            if photo_path is None:
                # PIL Image → 存为临时文件。
                # 用 mkstemp 拿到 fd 后立即 os.close，确保 Windows 下文件句柄不占用，
                # 再交给 PIL 写入，避免后续 Wav2Lip 读取时被句柄锁住。
                fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                image.save(tmp_name)
                photo_path = tmp_name

            try:
                voice = CHINESE_VOICES[voice_label]
                result = asyncio.run(run_pipeline(
                    photo_path=photo_path,
                    text=text,
                    voice=voice,
                    output_dir=str(PROJECT_ROOT / "output"),
                    enable_sadtalker=enable_sadtalker,
                    sadtalker_still=sadtalker_still,
                    enable_interpolation=enable_interpolation,
                ))
                return result["video"]
            finally:
                # 清理临时图片，避免每次生成残留文件
                if tmp_name and os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass

        btn.click(
            fn=process,
            inputs=[photo, text, voice, enable_sadtalker, sadtalker_still,
                    enable_interpolation],
            outputs=output_video,
        )

        # SadTalker 关闭时，"减少头部摆动"和"50fps 插帧"都没有意义 → 自动禁用
        def _toggle_sadtalker_options(enabled):
            return gr.update(interactive=enabled), gr.update(interactive=enabled)

        enable_sadtalker.change(
            fn=_toggle_sadtalker_options,
            inputs=enable_sadtalker,
            outputs=[sadtalker_still, enable_interpolation],
        )

        gr.Markdown("""
        ### 使用说明
        1. 上传一张清晰的人脸照片（正面、光照好、无遮挡）
        2. 输入你想让数字人说的话
        3. 选择语音风格，并勾选是否启用"自然表情与头部运动"
        4. 点击"生成视频"。CPU 模式下：开启 SadTalker 约十几分钟，关闭后约 1-3 分钟（均取决于视频长度）
        """)

    # 代理环境下 share=True 创建公网隧道，否则 localhost 检测会失败
    # 端口冲突时自动 +1 重试
    for port in range(7860, 7880):
        try:
            demo.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=True)
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) or "WinError 10048" in str(e):
                print(f"⚠️  端口 {port} 被占用，尝试 {port + 1}...")
                continue
            # 其他 OSError 不重试（比如 localhost 不可访问）
            print(f"⚠️  本地启动失败: {e}，尝试 share=True 公网模式...")
            demo.launch(server_name="0.0.0.0", server_port=port, share=True)
            break
    else:
        print("❌ 7860-7879 端口全部被占用，请手动关闭旧进程")
        print("   Windows: netstat -ano | findstr :7860   → taskkill /PID <pid> /F")


if __name__ == "__main__":
    main_cli()
