# coding: utf-8
"""
AutoDL 实例上的批量生成器：读 storyboard.json，逐段调 ComfyUI（H3 工作流）生成视频。

用法（在 remote_bundle 解压目录）：
  python run_segments.py                 # 全部段落
  python run_segments.py --only 3,4      # 只重跑指定段（失败重试场景）
  python run_segments.py --selftest      # 只校验工作流模板/补丁配置，不生成

流程：确保 ComfyUI 服务在 127.0.0.1:8188 → 按 h3_patch_map.json 把每段的
prompt / 参考图 / 参考音频 / 参考视频注入工作流 → POST /prompt → 轮询 /history
→ 产物复制到 out/segment_<id>.mp4。全部完成后可 export AUTODL_AUTO_SHUTDOWN=1 自动关机省租费。
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

BUNDLE = Path(__file__).parent.resolve()
COMFY_DIR = BUNDLE / "ComfyUI"
COMFY_URL = "http://127.0.0.1:8188"
CLIENT_ID = "sdpipeline"


def _http(method: str, path: str, payload: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(
        COMFY_URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _ensure_server(timeout: int = 600):
    """ComfyUI 没起就拉起子进程，等它就绪"""
    try:
        _http("GET", "/system_stats", timeout=5)
        print("✅ ComfyUI 已在运行")
        return None
    except Exception:
        pass
    print("🚀 启动 ComfyUI ...")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                            cwd=str(COMFY_DIR),
                            stdout=open(BUNDLE / "comfyui.log", "a"), stderr=subprocess.STDOUT)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _http("GET", "/system_stats", timeout=5)
            print("✅ ComfyUI 就绪")
            return proc
        except Exception:
            time.sleep(5)
    raise RuntimeError("ComfyUI 启动超时，请查看 comfyui.log")


def _load_graph_and_map():
    graph = json.loads((BUNDLE / "h3_workflow_api.json").read_text(encoding="utf-8"))
    pmap = json.loads((BUNDLE / "h3_patch_map.json").read_text(encoding="utf-8"))
    # 剥离下划线开头的说明键（_README 等），避免被当成节点/映射参与处理
    graph = {k: v for k, v in graph.items() if not k.startswith("_")}
    pmap = {k: v for k, v in pmap.items() if not k.startswith("_")}
    return graph, pmap


def _patch(graph: dict, pmap: dict, seg: dict, sb: dict):
    """按补丁映射把分镜字段写入工作流节点输入。
    pmap 形如：{"prompt": {"node": "6", "field": "text"},
               "image": {"node": "10", "field": "image"},
               "audio": {"node": "12", "field": "audio"},
               "video": {"node": "14", "field": "video"},
               "length": {"node": "20", "field": "length"}}
    """
    def set_in(node_key, value):
        loc = pmap.get(node_key)
        if not loc:
            return
        node = graph.get(loc["node"])
        if node is None:
            raise KeyError(f"补丁映射指向不存在的节点 {loc['node']}")
        node["inputs"][loc["field"]] = value

    def set_ref(node_key, value):
        """参考素材（图/音频/视频）必须先复制进 ComfyUI/input，节点里只写文件名"""
        loc = pmap.get(node_key)
        if not loc or not value:
            return
        src = BUNDLE / value
        if src.exists():
            input_dir = COMFY_DIR / "input"
            input_dir.mkdir(exist_ok=True)
            shutil.copyfile(src, input_dir / src.name)
            value = src.name
        node = graph.get(loc["node"])
        if node is None:
            raise KeyError(f"补丁映射指向不存在的节点 {loc['node']}")
        node["inputs"][loc["field"]] = value

    set_in("prompt", seg.get("prompt", ""))
    set_in("length", int(seg.get("seconds", 15)))
    if seg.get("kind") == "char_intro" or seg.get("image"):
        # 角色参考图（虚拟人物用定妆帧，真人用户可直接给照片）
        set_ref("image", seg.get("image") or sb.get("char_image"))
    if seg.get("kind") in ("sing",) and sb.get("song"):
        set_ref("audio", sb["song"])
    if seg.get("kind") == "dance" and sb.get("video_ref"):
        set_ref("video", sb["video_ref"])
    return graph


def _generate_one(graph: dict, seg_id: int, out_dir: Path):
    job = _http("POST", "/prompt", {"prompt": graph, "client_id": CLIENT_ID})
    prompt_id = job["prompt_id"]
    print(f"   段#{seg_id:03d} 已提交 {prompt_id}，生成中（几分钟）...")
    deadline = time.time() + 1800
    while time.time() < deadline:
        time.sleep(10)
        hist = _http("GET", f"/history/{prompt_id}")
        if prompt_id not in hist:
            continue
        entry = hist[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"段#{seg_id:03d} 生成失败：{json.dumps(status)[:400]}")
        outputs = entry.get("outputs", {})
        files = []
        for node_out in outputs.values():
            for key in ("gifs", "videos", "images"):
                for item in node_out.get(key, []):
                    files.append(item)
        if files:
            src = (COMFY_DIR / "output" / files[0]["filename"])
            dst = out_dir / f"segment_{seg_id:03d}.mp4"
            if src.suffix.lower() != ".mp4":  # 部分节点出 webm/gif，统一转 mp4
                subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264",
                                "-crf", "18", str(dst)], capture_output=True, check=True)
            else:
                shutil.copyfile(src, dst)
            print(f"   ✅ 段#{seg_id:03d} → {dst.name}")
            return
    raise TimeoutError(f"段#{seg_id:03d} 超时")


def selftest():
    graph, pmap = _load_graph_and_map()
    assert isinstance(graph, dict) and graph, "工作流模板为空"
    for key, loc in pmap.items():
        node = graph.get(loc["node"])
        assert node is not None, f"补丁映射 {key} 指向不存在的节点 {loc['node']}"
        assert loc["field"] in node.get("inputs", {}), \
            f"补丁映射 {key}: 节点 {loc['node']} 没有 inputs.{loc['field']}"
    sb = json.loads((BUNDLE / "storyboard.json").read_text(encoding="utf-8"))
    seg = (sb.get("segments") or [{}])[0]
    _patch(json.loads(json.dumps(graph)), pmap, seg, sb)
    print("✅ selftest 通过：工作流/补丁映射/分镜三者字段对得上")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只生成指定段，如 3,4")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    graph, pmap = _load_graph_and_map()
    sb = json.loads((BUNDLE / "storyboard.json").read_text(encoding="utf-8"))
    out_dir = BUNDLE / "out"
    out_dir.mkdir(exist_ok=True)

    proc = _ensure_server()
    only = {int(x) for x in args.only.split(",")} if args.only else None
    ok, fail = 0, []
    for seg in sb["segments"]:
        sid = int(seg["id"])
        if only and sid not in only:
            continue
        if (out_dir / f"segment_{sid:03d}.mp4").exists():
            print(f"   ⏭  段#{sid:03d} 已存在，跳过（失败重试删掉对应文件）")
            continue
        g = _patch(json.loads(json.dumps(graph)), pmap, seg, sb)
        try:
            _generate_one(g, sid, out_dir)
            ok += 1
        except Exception as e:
            print(f"   ❌ 段#{sid:03d}: {e}")
            fail.append(sid)

    shutil.make_archive(str(out_dir / "segments"), "zip", root_dir=out_dir)
    print(f"📦 完成：成功 {ok} 段，失败 {len(fail)} 段 {fail or ''}")
    print(f"   下载 out/segments.zip 回本机后解压到 output/sd_projects/<项目>/segments/ 即可 assemble")
    if proc:
        proc.terminate()
    if os.environ.get("AUTODL_AUTO_SHUTDOWN") == "1":
        print("🛑 AUTODL_AUTO_SHUTDOWN=1，5 秒后自动关机停止计费...")
        time.sleep(5)
        os.system("shutdown")


if __name__ == "__main__":
    main()
