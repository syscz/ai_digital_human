# coding: utf-8
"""
唱跳视频流水线（H3 开源权重 + AutoDL 租卡 GPU）
================================================
输入：一句话主题（可选：歌曲音频 / 歌词 / 舞蹈参考视频 / 角色图）
输出：约 2 分钟的唱跳成片 MP4

三步走：
  1) storyboard  本机 Ollama 生成分镜脚本（--no-llm 用内置模板，不依赖 LLM）
  2) pack        打包"远程执行包"（分镜+素材+脚本）→ 上传 AutoDL 实例执行
  3) assemble    生成好的分段视频取回本机 → ffmpeg 拼接+音轨+字幕 → 成片

成本参考：4090 约 ¥2/小时，一条 2 分钟成片（8~10 段）GPU 时间约 40 分钟 ≈ ¥1~2。
本机只承担 LLM 分镜和 ffmpeg 拼接两件轻活，不会触发满载断电。

典型用法：
  python song_dance_pipeline.py storyboard --theme "银发赛博朋克女歌手的舞台表演" --song song.mp3
  python song_dance_pipeline.py pack       --project mymv
  python song_dance_pipeline.py assemble   --project mymv
（项目目录：output/sd_projects/<name>/，分镜/素材/分段/成片都在里面）
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
H3_REMOTE_DIR = PROJECT_ROOT / "h3_remote"          # 随包分发的远程脚本模板
PROJECTS_DIR = PROJECT_ROOT / "output" / "sd_projects"
MAX_SEG_SECONDS = 15                                 # H3 单段上限
DEFAULT_SIZE = "1080x1920"                           # 竖屏成片尺寸


def get_ffmpeg_exe() -> str:
    local = PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe"
    return str(local) if local.exists() else "ffmpeg"


def _probe_duration(path: str) -> float:
    """用 cv2 探测视频时长（imageio-ffmpeg 不带 ffprobe，避免依赖）"""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps and frames else 0.0


def _read_lyrics(path: str | None) -> list[str]:
    if not path:
        return []
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines


# ========== Step 1: 分镜 ==========
_FALLBACK_EN = {
    "char_intro": "Full-body shot of {theme}, stage spotlight, slow camera push-in, cinematic",
    "sing": "Close-up of {theme} singing passionately, rim light, particles in the air",
    "dance": "{theme} performing energetic dance moves on stage, dynamic camera, neon lights",
    "transition": "Wide shot of the stage and crowd silhouettes, light beams sweeping, atmospheric",
}


def _ollama_storyboard(theme: str, lyrics: list[str], n_segments: int) -> dict | None:
    """调本机 Ollama 生成分镜 JSON；失败返回 None（调用方回退模板）"""
    model = os.environ.get("SD_LLM_MODEL", "qwen2.5:7b")
    sys_prompt = (
        "你是 MV 分镜脚本师。根据主题输出严格的 JSON（不要多余文字），结构："
        '{"title": str, "character_desc": str(虚拟人物外形设定，中文), '
        '"segments": [{"kind": "char_intro|sing|dance|transition", "seconds": 10~15的整数, '
        '"prompt": str(英文画面描述，风格统一、适合视频生成模型), "lyric": str(该段中文字幕，可为空)}]}。'
        f"要求：恰好 {n_segments} 段；第1段是 char_intro；sing 和 dance 段交替出现；"
        "prompt 用英文、细节丰富（构图/光线/运镜）。"
    )
    user = f"主题：{theme}"
    if lyrics:
        user += "\n歌词（请按顺序分配到 sing 段的 lyric 字段）：\n" + "\n".join(lyrics)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": user}],
        "format": "json", "stream": False,
        "options": {"temperature": 0.7},
    }).encode("utf-8")
    try:
        req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode("utf-8"))
        sb = json.loads(data["message"]["content"])
    except Exception as e:
        print(f"   ⚠️  Ollama 不可用（{e}），回退内置模板分镜")
        return None
    segs = sb.get("segments") or []
    if not (4 <= len(segs) <= 14):
        print("   ⚠️  LLM 分镜段数异常，回退内置模板")
        return None
    return sb


def cmd_storyboard(args):
    lyrics = _read_lyrics(args.lyrics)
    n = args.segments
    sb = None
    if not args.no_llm:
        print(f"🧠  调用 Ollama 生成分镜（模型 {os.environ.get('SD_LLM_MODEL', 'qwen2.5:7b')}）...")
        sb = _ollama_storyboard(args.theme, lyrics, n)
    if sb is None:
        # 内置模板：intro → sing/dance 交替 → transition 收尾
        kinds = ["char_intro"] + (["sing", "dance"] * n)[: n - 2] + ["transition"]
        per_lyric = max(1, len(lyrics) // max(1, sum(1 for k in kinds if k == "sing"))) if lyrics else 0
        li = 0
        segs = []
        for kind in kinds:
            lyric = ""
            if kind == "sing" and lyrics:
                chunk = lyrics[li:li + per_lyric] or [""]
                lyric = " / ".join(chunk)
                li += per_lyric
            segs.append({"kind": kind, "seconds": 15,
                         "prompt": _FALLBACK_EN[kind].format(theme=args.theme),
                         "lyric": lyric})
        sb = {"title": args.theme[:24], "character_desc": args.theme, "segments": segs}

    for i, seg in enumerate(sb["segments"], 1):
        seg["id"] = i
        seg["seconds"] = int(min(max(seg.get("seconds", 15), 8), MAX_SEG_SECONDS))
        seg.setdefault("kind", "sing")
        seg.setdefault("prompt", "")
        seg.setdefault("lyric", "")

    pdir = PROJECTS_DIR / args.project
    pdir.mkdir(parents=True, exist_ok=True)
    sb["theme"] = args.theme
    out = pdir / "storyboard.json"
    out.write_text(json.dumps(sb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 分镜完成：{len(sb['segments'])} 段 → {out}")
    for s in sb["segments"]:
        print(f"   #{s['id']:02d} [{s['kind']:<10s}] {s['seconds']}s  {s['lyric'] or s['prompt'][:40]}")


# ========== Step 2: 打包远程执行包 ==========
def cmd_pack(args):
    pdir = PROJECTS_DIR / args.project
    sb_path = pdir / "storyboard.json"
    if not sb_path.exists():
        sys.exit(f"❌ 找不到 {sb_path}，请先执行 storyboard 子命令")
    sb = json.loads(sb_path.read_text(encoding="utf-8"))

    bundle = pdir / "remote_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "assets").mkdir(parents=True)

    # 素材：歌曲 / 舞蹈参考 / 角色图
    for key, arg in (("song", args.song), ("dance_ref", args.dance_ref), ("char_image", args.char_image)):
        if arg:
            if not Path(arg).exists():
                sys.exit(f"❌ 素材不存在：{arg}")
            shutil.copyfile(arg, bundle / "assets" / Path(arg).name)
            sb[key] = f"assets/{Path(arg).name}"
    sb_path.write_text(json.dumps(sb, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(sb_path, bundle / "storyboard.json")

    # 远程脚本（随包分发）
    for name in ("setup_h3.sh", "download_h3_models.py", "run_segments.py",
                 "h3_workflow_api.json", "h3_patch_map.json", "README_REMOTE.txt"):
        shutil.copyfile(H3_REMOTE_DIR / name, bundle / name)

    zip_base = pdir / "remote_bundle"
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=bundle)
    size = os.path.getsize(archive) / 1024 / 1024
    print(f"✅ 远程执行包：{archive} ({size:.1f} MB)")
    print("   下一步：按 h3_remote/README_REMOTE.txt 上传 AutoDL 执行，完成后把 out/segments.zip 取回")


# ========== Step 3: 拼装成片 ==========
def cmd_assemble(args):
    pdir = PROJECTS_DIR / args.project
    sb = json.loads((pdir / "storyboard.json").read_text(encoding="utf-8"))
    seg_dir = pdir / "segments"
    segs = sorted(seg_dir.glob("segment_*.mp4"))
    if len(segs) < 2:
        sys.exit(f"❌ {seg_dir} 下分段视频不足 2 个，请先完成远程生成并解压到该目录")

    W, H = (int(x) for x in args.size.split("x"))
    work = pdir / "assemble_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1) 归一化：统一分辨率/帧率/编码，去掉各段自带的杂乱音轨
    print(f"🎬  归一化 {len(segs)} 段 → {W}x{H}@30fps ...")
    norms = []
    for seg in segs:
        norm = work / f"norm_{seg.stem}.mp4"
        subprocess.run([get_ffmpeg_exe(), "-y", "-i", str(seg),
                        "-vf", (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p"),
                        "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                        str(norm)], capture_output=True, check=True)
        norms.append(norm)

    # 2) 无缝拼接（已归一化，concat 直接拷贝流）
    list_file = work / "concat.txt"
    list_file.write_text("".join(f"file '{n.as_posix()}'\n" for n in norms), encoding="utf-8")
    joined = work / "joined.mp4"
    subprocess.run([get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", str(joined)], capture_output=True, check=True)

    # 3) 混入歌曲作为主音轨（唱跳场景以原曲为准；H3 生成的段内音频仅作参考）
    final = pdir / "final.mp4"
    song = None
    if sb.get("song"):
        # pack 时素材被复制进 remote_bundle/assets/，两处都找一下
        for cand in (pdir / sb["song"], pdir / "remote_bundle" / sb["song"]):
            if cand.exists():
                song = cand
                break
    if song and Path(song).exists():
        subprocess.run([get_ffmpeg_exe(), "-y", "-i", str(joined), "-i", str(song),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)],
                       capture_output=True, check=True)
    else:
        print("   ⚠️  未提供歌曲文件，成片无主音轨")
        shutil.copyfile(joined, final)

    # 4) 字幕：按分镜 lyric 生成 SRT 副本（软字幕随片分发，避免依赖 libass）
    def _ts(sec):
        h, m = int(sec // 3600), int(sec % 3600 // 60)
        return f"{h:02d}:{m:02d}:{sec % 60:05.2f}"

    srt_lines, t0 = [], 0.0
    for i, n in enumerate(norms):
        lyric = (sb["segments"][i].get("lyric") or "").strip() if i < len(sb["segments"]) else ""
        d = _probe_duration(str(n))
        if lyric:
            srt_lines.append(f"{i + 1}\n{_ts(t0)} --> {_ts(t0 + d)}\n{lyric}\n")
        t0 += d
    if srt_lines:
        srt = pdir / "final.srt"
        srt.write_text("\n".join(srt_lines), encoding="utf-8")
        print(f"   💬 字幕副本：{srt}")

    size = os.path.getsize(final) / 1024 / 1024
    print(f"🎉  成片完成：{final} ({_probe_duration(str(final)):.1f}s, {size:.1f} MB)")


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(description="唱跳视频流水线（H3 + AutoDL）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("storyboard", help="生成分镜脚本")
    p.add_argument("--theme", required=True, help="一句话主题，如：银发赛博朋克女歌手的舞台表演")
    p.add_argument("--project", required=True, help="项目名（output/sd_projects/<名>/）")
    p.add_argument("--lyrics", default=None, help="歌词文本文件（可选，逐行）")
    p.add_argument("--segments", type=int, default=9, help="分镜段数（默认 9，每段最长 15s）")
    p.add_argument("--no-llm", action="store_true", help="跳过 Ollama，用内置模板分镜")

    p = sub.add_parser("pack", help="打包远程执行包")
    p.add_argument("--project", required=True)
    p.add_argument("--song", default=None, help="歌曲音频文件（mp3/wav）")
    p.add_argument("--dance-ref", default=None, help="舞蹈参考视频（可选，跳舞段用）")
    p.add_argument("--char-image", default=None, help="角色参考图（可选；不传则按描述生成虚拟人物）")

    p = sub.add_parser("assemble", help="拼接成片")
    p.add_argument("--project", required=True)
    p.add_argument("--size", default=DEFAULT_SIZE, help=f"成片尺寸（默认 {DEFAULT_SIZE}）")

    args = parser.parse_args()
    {"storyboard": cmd_storyboard, "pack": cmd_pack, "assemble": cmd_assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
