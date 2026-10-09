# coding: utf-8
"""AutoDL 远程：下载 MiniMax H3 三组件到 ComfyUI/models 对应目录。

仓库结构以 Comfy-Org 重打包为准（diffusion_models / text_encoders / vae），
仓名可用 --repo 或环境变量 H3_MODEL_REPO 覆盖；优先 huggingface，失败自动切 ModelScope。
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("H3_MODEL_REPO", "Comfy-Org/MiniMax-H3_repackaged"))
    ap.add_argument("--base", default="ComfyUI/models", help="ComfyUI models 根目录")
    args = ap.parse_args()

    # 目标子目录：ComfyUI 约定
    targets = {
        "diffusion_models": os.path.join(args.base, "diffusion_models"),
        "text_encoders": os.path.join(args.base, "text_encoders"),
        "vae": os.path.join(args.base, "vae"),
    }
    for d in targets.values():
        os.makedirs(d, exist_ok=True)

    def try_download(lib: str) -> bool:
        try:
            if lib == "hf":
                from huggingface_hub import snapshot_download
                path = snapshot_download(
                    repo_id=args.repo,
                    allow_patterns=["*.safetensors", "*.json"],
                    max_workers=4,
                )
            else:
                from modelscope import snapshot_download
                path = snapshot_download(args.repo)
            print(f"已下载到: {path}")
            # 按文件名关键词归位到 models 子目录
            import shutil, glob
            for f in glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True):
                name = os.path.basename(f).lower()
                key = ("text_encoders" if ("text" in name or "t5" in name or "clip" in name)
                       else "vae" if "vae" in name
                       else "diffusion_models")
                dst = os.path.join(targets[key], os.path.basename(f))
                if not os.path.exists(dst):
                    print(f"  {os.path.basename(f)} -> {key}/")
                    shutil.copyfile(f, dst)
            return True
        except Exception as e:
            print(f"{lib} 下载失败：{e}")
            return False

    if try_download("hf") or try_download("ms"):
        print("✅ H3 模型组件就绪")
    else:
        sys.exit("❌ 两个源都失败：请检查 H3_MODEL_REPO 是否为正确的 Comfy-Org 重打包仓，"
                 "或参考 ComfyUI 官方 H3 模板页面的下载说明手动放置文件")


if __name__ == "__main__":
    main()
