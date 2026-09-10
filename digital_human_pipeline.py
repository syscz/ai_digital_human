# coding: utf-8
"""
AI 数字人端到端流水线
=====================
输入：源图片 + 文本
输出：说话视频（嘴型跟音频同步）

流水线：文本 → Edge-TTS(语音合成) → WAV → Wav2Lip(音频驱动) → MP4

使用方式：
  # 命令行
  python digital_human_pipeline.py --photo photo.jpg --text "你好，我是数字人" --voice zh-CN-XiaoxiaoNeural --output result.mp4

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
WAV2LIP_CHECKPOINT = WAV2LIP_DIR / "checkpoints" / "wav2lip.pth"
WAV2LIP_TEMP = WAV2LIP_DIR / "temp"
CONDA_ENV = "LivePortrait"

# 常用中文语音
CHINESE_VOICES = {
    "女声-亲切": "zh-CN-XiaoxiaoNeural",
    "女声-温柔": "zh-CN-XiaoyiNeural",
    "男声-新闻": "zh-CN-YunxiNeural",
    "男声-年轻": "zh-CN-YunjianNeural",
    "男声-稳重": "zh-CN-YunyangNeural",
}


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

    # 检查 ffmpeg
    ffmpeg_ok = False
    # 先在 PATH 找
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    if r.returncode == 0:
        ffmpeg_ok = True
    # 再在 Wav2Lip/ffmpeg 找
    local_ffmpeg = PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        ffmpeg_ok = True
        os.environ["PATH"] = str(local_ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")
    if not ffmpeg_ok:
        issues.append("❌ FFmpeg 未安装或不在 PATH 中")

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
        "ffmpeg", "-y",
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
    photo_path: str,
    audio_path: str,
    output_path: str,
    wav2lip_checkpoint: str = None,
    batch_size: int = 8,
) -> str:
    """
    用 Wav2Lip 把照片 + 音频合成为说话视频

    Args:
        photo_path: 源图片路径
        audio_path: 音频文件路径（wav）
        output_path: 输出视频路径（mp4）
        wav2lip_checkpoint: Wav2Lip 模型路径
        batch_size: 推理 batch size（CPU 建议 8，GPU 可 128）

    Returns:
        生成的视频文件路径
    """
    if wav2lip_checkpoint is None:
        wav2lip_checkpoint = str(WAV2LIP_CHECKPOINT)

    # 确保 Wav2Lip/temp/ 存在（inference.py 里硬编码用这个目录）
    WAV2LIP_TEMP.mkdir(parents=True, exist_ok=True)

    print(f"🎬  Wav2Lip 音频驱动...")
    print(f"   照片: {photo_path}")
    print(f"   音频: {audio_path}")
    print(f"   模型: {wav2lip_checkpoint}")
    print(f"   Batch size: {batch_size}")

    cmd = [
        sys.executable, str(WAV2LIP_INFERENCE),
        "--checkpoint_path", wav2lip_checkpoint,
        "--face", photo_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--static", "True",          # 源是图片，只用第一帧
        "--pads", "0", "15", "0", "0",  # 底部多 padding 防止切到下巴
        "--wav2lip_batch_size", str(batch_size),
        "--nosmooth",                # 静态图不需要时间平滑
    ]

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


# ========== 主流水线 ==========
async def run_pipeline(
    photo_path: str,
    text: str,
    voice: str = "zh-CN-XiaoxiaoNeural",
    output_dir: str = None,
    wav2lip_checkpoint: str = None,
) -> dict:
    """
    端到端流水线：文本 + 照片 → 说话视频

    Args:
        photo_path: 源图片路径
        text: 要说的话
        voice: Edge-TTS 语音名称
        output_dir: 输出目录（默认在 output/ 下）
        wav2lip_checkpoint: Wav2Lip 模型路径（默认用 wav2lip.pth）

    Returns:
        dict: {"audio": "xxx.wav", "video": "xxx.mp4"}
    """
    # 准备输出目录
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "output")
    os.makedirs(output_dir, exist_ok=True)

    # 用时间戳避免文件名冲突
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(output_dir, f"speech_{timestamp}.wav")
    video_path = os.path.join(output_dir, f"talking_{timestamp}.mp4")

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

        # Step 2: 音频驱动
        audio_to_video(photo_path, audio_path, video_path, wav2lip_checkpoint)

        print("\n" + "=" * 60)
        print(f"🎉 流水线完成！")
        print(f"   音频: {audio_path}")
        print(f"   视频: {video_path}")
        print("=" * 60)

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
    parser.add_argument("--output", default=None, help="输出视频路径 (默认 output/talking_时间戳.mp4)")
    parser.add_argument("--checkpoint", default=None, help="Wav2Lip 模型路径 (默认 checkpoints/wav2lip.pth)")
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
        output_dir=str(PROJECT_ROOT / "output"),
        wav2lip_checkpoint=args.checkpoint,
    ))

    print(f"\n🎬 生成的视频: {result['video']}")


# ========== Gradio 入口 ==========
def launch_gradio():
    try:
        import gradio as gr
    except ImportError:
        print("请先安装 gradio: conda activate LivePortrait && pip install gradio")
        sys.exit(1)

    def process(photo, text, voice_label):
        """Gradio 处理函数"""
        voice = CHINESE_VOICES[voice_label]
        result = asyncio.run(run_pipeline(
            photo_path=photo,
            text=text,
            voice=voice,
            output_dir=str(PROJECT_ROOT / "output"),
        ))
        return result["video"]

    with gr.Blocks(title="AI 数字人 - 文本生成说话视频") as demo:
        gr.Markdown("# 🎬 AI 数字人\n上传一张照片 + 输入文字，生成说话视频（Wav2Lip + Edge-TTS）")

        with gr.Row():
            with gr.Column():
                photo = gr.Image(label="📷 源图片", type="filepath")
                text = gr.Textbox(label="📝 要说的话", value="你好，我是数字人，很高兴见到你！", lines=3)
                voice = gr.Dropdown(
                    label="🎙️ 语音选择",
                    choices=list(CHINESE_VOICES.keys()),
                    value="女声-亲切",
                )
                btn = gr.Button("🚀 生成视频", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="🎥 生成的说话视频")

        btn.click(
            fn=process,
            inputs=[photo, text, voice],
            outputs=output_video,
        )

        gr.Markdown("""
        ### 使用说明
        1. 上传一张清晰的人脸照片（正面、光照好、无遮挡）
        2. 输入你想让数字人说的话
        3. 选择语音风格
        4. 点击"生成视频"（CPU 模式约需 20-60 秒，取决于视频长度）
        """)

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main_cli()
