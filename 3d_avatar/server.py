# coding: utf-8
"""
3D 数字人后端服务器
===================
提供：
  1. 静态文件服务（Three.js 页面、Avatar GLB、Mixamo FBX、audio 目录）
  2. /api/files — 列出可用的 Avatar 和动画文件
  3. /api/tts — 用 Edge-TTS 生成语音（返回 URL）

启动：
  conda activate LivePortrait
  cd d:\\Work\\Python\\ai_digital_human\\3d_avatar
  python server.py
  # 浏览器打开 http://localhost:8080
"""

import asyncio
import json
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "audio"
OUTPUT_DIR.mkdir(exist_ok=True)

HOST = "0.0.0.0"
PORT = 8080


class DigitalHumanHandler(SimpleHTTPRequestHandler):
    """自定义 Handler：API 路由 + 静态文件"""

    def end_headers(self):
        # CORS — 方便直接用 file:// 打开 HTML 时也能调 API
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/files":
            return self.handle_files()
        elif path == "/api/tts":
            # TTS 也支持 GET（方便测试）
            params = parse_qs(parsed.query)
            text = params.get("text", [""])[0]
            voice = params.get("voice", ["zh-CN-XiaoxiaoNeural"])[0]
            return self.handle_tts(text, voice)
        else:
            # 静态文件 — 确保 CWD 正确
            return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/tts":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {}
            text = data.get("text", "")
            voice = data.get("voice", "zh-CN-XiaoxiaoNeural")
            return self.handle_tts(text, voice)
        else:
            self.send_response(404)
            self.end_headers()

    # ========== API 实现 ==========

    def handle_files(self):
        """列出 avatar/ 和 animations/ 目录下的文件"""
        def list_dir(rel_path, extensions):
            folder = PROJECT_ROOT / rel_path
            if not folder.exists():
                return []
            return sorted(
                f"{rel_path}/{f.name}"
                for f in folder.iterdir()
                if f.suffix.lower() in extensions
            )

        result = {
            "avatars": list_dir("avatar", {".glb", ".gltf", ".vrm", ".fbx"}),
            "animations": list_dir("animations", {".fbx", ".glb", ".bvh"}),
            "audio": list_dir("audio", {".mp3", ".wav", ".ogg"}),
        }

        self.send_json(200, result)

    def handle_tts(self, text: str, voice: str):
        """Edge-TTS 生成语音"""
        if not text.strip():
            self.send_json(400, {"error": "text is required"})
            return

        try:
            import edge_tts
        except ImportError:
            self.send_json(500, {"error": "edge-tts not installed. Run: pip install edge-tts"})
            return

        # 生成唯一文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(c if c.isalnum() or c == " " else "_" for c in text)[:20]
        out_name = f"tts_{timestamp}_{safe_text}.mp3"
        out_path = OUTPUT_DIR / out_name
        out_url = f"/audio/{out_name}"

        print(f"🎙️  TTS: voice={voice} text='{text[:50]}'...")

        # 同步调用 async
        try:
            asyncio.run(edge_tts.Communicate(text=text, voice=voice).save(str(out_path)))
        except Exception as e:
            self.send_json(500, {"error": f"TTS generation failed: {e}"})
            return

        self.send_json(200, {
            "url": out_url,
            "file": out_name,
            "size": out_path.stat().st_size if out_path.exists() else 0,
        })

    # ========== 辅助 ==========

    def send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    # 确保 CWD 是 3d_avatar 目录（SimpleHTTPRequestHandler 相对此提供服务）
    os.chdir(str(PROJECT_ROOT))

    server = HTTPServer((HOST, PORT), DigitalHumanHandler)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎬 3D 数字人后端已启动                                       ║
║                                                              ║
║  前端页面:   http://localhost:{PORT}/index.html              ║
║  API 测试:   http://localhost:{PORT}/api/files               ║
║                                                              ║
║  目录说明:                                                    ║
║    avatar/      ← 放 Ready Player Me 导出的 .glb             ║
║    animations/  ← 放 Mixamo 下载的 .fbx 动作                 ║
║    audio/       ← TTS 生成的音频会自动放这里                   ║
║                                                              ║
║  停止: Ctrl+C                                                ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
