#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemorMe 一键部署脚本 (跨平台 Python)
======================================
完成: 环境检查 → 虚拟环境 → 依赖安装 → 模型下载 → MCP 配置写入 → 验证

用法:
  python deploy.py                          # 默认部署 jina-v3，CPU
  python deploy.py --model bge-m3           # 指定模型
  python deploy.py --gpu                    # CUDA 加速 (自动检测版本)
  python deploy.py --gpu --cuda cu121       # 指定 CUDA 版本
  python deploy.py --skip-download          # 跳过模型下载
  python deploy.py --skip-config            # 跳过 MCP 配置写入
  python deploy.py --uninstall              # 删除 .venv
  python deploy.py --download-only          # 仅下载模型 (不建虚拟环境)
  python deploy.py --list-models            # 列出可用模型

示例:
  # 首次安装 (推荐)
  python deploy.py --model jina-v3 --gpu

  # 低配机器
  python deploy.py --model minilm

  # 只换模型, 其他不动
  python deploy.py --download-only --model bge-m3
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
import threading
from pathlib import Path
from typing import Optional

# ── Windows 终端 UTF-8 ──────────────────────────────────────────────
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 常量 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

MODELS: dict[str, dict] = {
    "jina-v3": {
        "hf_name":       "jinaai/jina-embeddings-v3",
        "size_mb":       570,
        "dim":           1024,
        "max_seq":       8192,
        "late_chunking": True,
        "trust_remote":  True,
        "desc":          "推荐 · 长上下文 · Late Chunking · 多语言",
    },
    "bge-m3": {
        "hf_name":       "BAAI/bge-m3",
        "size_mb":       2200,
        "dim":           1024,
        "max_seq":       8192,
        "late_chunking": False,
        "trust_remote":  False,
        "desc":          "最高精度 · 100+ 语言",
    },
    "bge-small-zh": {
        "hf_name":       "BAAI/bge-small-zh-v1.5",
        "size_mb":       95,
        "dim":           512,
        "max_seq":       512,
        "late_chunking": False,
        "trust_remote":  False,
        "desc":          "轻量 · 中文",
    },
    "bge-small-en": {
        "hf_name":       "BAAI/bge-small-en-v1.5",
        "size_mb":       130,
        "dim":           384,
        "max_seq":       512,
        "late_chunking": False,
        "trust_remote":  False,
        "desc":          "轻量 · 英文",
    },
    "minilm": {
        "hf_name":       "all-MiniLM-L6-v2",
        "size_mb":       90,
        "dim":           384,
        "max_seq":       256,
        "late_chunking": False,
        "trust_remote":  False,
        "desc":          "超轻量 · 英文回退",
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  终端 UI 工具
# ═══════════════════════════════════════════════════════════════════════
def _tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code: int, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty() else text

def step(msg: str)  -> None: print(_c(36, f"\n▶ {msg}"))       # cyan
def ok(msg: str)    -> None: print(_c(32, f"  ✔ {msg}"))       # green
def warn(msg: str)  -> None: print(_c(33, f"  ⚠ {msg}"))       # yellow
def fail(msg: str)  -> None: print(_c(31, f"  ✘ {msg}"))       # red
def dim(msg: str)   -> str:  return _c(2, msg)
def bold(msg: str)  -> str:  return _c(1, msg)

def banner(lines: list[str]) -> None:
    sep = "━" * 52
    print(_c(32, f"\n{sep}"))
    for ln in lines:
        print(_c(32, f"  {ln}"))
    print(_c(32, sep))
    print()


# ═══════════════════════════════════════════════════════════════════════
#  核心操作
# ═══════════════════════════════════════════════════════════════════════


def _run(cmd: list[str], label: str, **kw) -> subprocess.CompletedProcess:
    """运行子进程，失败时报错退出。"""
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)
        return r
    except subprocess.CalledProcessError as e:
        fail(f"{label} 失败 (exit {e.returncode})")
        if e.stderr:
            for line in e.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
        sys.exit(1)
    except FileNotFoundError:
        fail(f"找不到命令: {cmd[0]}")
        sys.exit(1)


def _pip(venv: Path) -> str:
    """返回虚拟环境中的 pip 可执行路径。"""
    if sys.platform == "win32":
        return str(venv / "Scripts" / "pip.exe")
    return str(venv / "bin" / "pip")


def _python(venv: Path) -> str:
    """返回虚拟环境中的 python 可执行路径。"""
    if sys.platform == "win32":
        return str(venv / "Scripts" / "python.exe")
    return str(venv / "bin" / "python")


# ── 1. 检查 Python ───────────────────────────────────────────────────
def check_python() -> str:
    step("检查 Python 版本")
    ver = sys.version_info
    if ver < (3, 10):
        fail(f"需要 Python >= 3.10，当前: {ver.major}.{ver.minor}")
        sys.exit(1)
    py_exec = sys.executable
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}  ({py_exec})")
    return py_exec


# ── 2. 创建虚拟环境 ──────────────────────────────────────────────────
def create_venv(venv: Path) -> None:
    step("创建虚拟环境 (.venv)")
    if venv.exists():
        warn(f".venv 已存在，跳过创建  ({venv})")
        return
    import venv as venv_mod
    venv_mod.create(str(venv), with_pip=True)
    ok(f"虚拟环境创建完成  ({venv})")


# ── 3. 升级 pip ──────────────────────────────────────────────────────
def upgrade_pip(venv: Path) -> None:
    step("升级 pip")
    _run([_python(venv), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
         "pip upgrade")
    ok("pip 已升级")


# ── 4. 安装依赖 ──────────────────────────────────────────────────────
def install_deps(venv: Path, gpu: bool, cuda_ver: str) -> None:
    pip = _pip(venv)

    step("安装项目依赖")
    _run([pip, "install", "-e", f"{ROOT}.[hnsw]", "--quiet"], "pip install -e .[hnsw]")
    ok("项目包安装完成  (.[hnsw])")

    step("安装 transformers + torch")
    if gpu:
        warn(f"安装 CUDA ({cuda_ver}) 版 torch，文件较大…")
        _run([pip, "install", "torch", "torchvision",
              "--index-url", f"https://download.pytorch.org/whl/{cuda_ver}",
              "--quiet"], "torch (CUDA)")
        ok(f"torch (CUDA {cuda_ver}) 安装完成")
    else:
        _run([pip, "install", "transformers>=4.40.0", "torch", "--quiet"],
             "transformers + torch")
        # 检测 Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            ok("transformers + torch (MPS) 安装完成")
        else:
            ok("transformers + torch (CPU) 安装完成")


# ── 5. 下载模型 ──────────────────────────────────────────────────────
def _spinner(stop: threading.Event) -> None:
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not stop.is_set():
        sys.stdout.write(f"\r  {_c(33, chars[i % len(chars)])} 下载中…")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


def download_model(model_key: str, venv: Optional[Path] = None,
                   verify: bool = True) -> bool:
    m = MODELS[model_key]
    step(f"下载模型: {bold(model_key)}  (~{m['size_mb']} MB)")
    print(f"  HuggingFace : {m['hf_name']}")
    print(f"  维度 {m['dim']}  |  最大 Token {m['max_seq']}  |  "
          f"Late Chunking {'✅' if m['late_chunking'] else '—'}")
    warn("首次从 HuggingFace 下载，网速较慢时请耐心等待…")

    # 用虚拟环境的 python 运行，保证 sentence_transformers 可用
    py = _python(venv) if venv else sys.executable
    trust_arg = "True" if m["trust_remote"] else "False"

    dl_script = textwrap.dedent(f"""\
        import time, sys
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        kwargs = {{"trust_remote_code": True}} if {trust_arg} else {{}}
        model = SentenceTransformer("{m['hf_name']}", **kwargs)
        dim = model.get_sentence_embedding_dimension()
        elapsed = time.time() - t0
        print(f"OK dim={{dim}} elapsed={{elapsed:.1f}}s")
    """)

    stop_evt = threading.Event()
    if _tty():
        t = threading.Thread(target=_spinner, args=(stop_evt,), daemon=True)
        t.start()

    try:
        r = subprocess.run(
            [py, "-c", dl_script],
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        stop_evt.set()
        fail("模型下载超时 (>30 min)，请检查网络")
        return False
    finally:
        stop_evt.set()

    if r.returncode != 0:
        fail(f"模型下载失败")
        for line in (r.stderr or "").strip().splitlines()[-6:]:
            print(f"    {line}")
        warn("提示: 检查网络，或设置 HF_ENDPOINT 环境变量使用镜像站")
        return False

    # 解析输出
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if out.startswith("OK "):
        parts = dict(kv.split("=") for kv in out[3:].split())
        ok(f"下载完成  维度 {parts.get('dim','')}  耗时 {parts.get('elapsed','')}s")
    else:
        ok("下载完成")

    # 验证推理
    if verify:
        _verify_inference(py, m)

    return True


def _verify_inference(py: str, m: dict) -> None:
    verify_script = textwrap.dedent(f"""\
        from sentence_transformers import SentenceTransformer
        import numpy as np, time
        kwargs = {{"trust_remote_code": True}} if {m['trust_remote']} else {{}}
        model = SentenceTransformer("{m['hf_name']}", **kwargs)
        texts = ["MemorMe 记忆系统测试", "This is a test sentence."]
        t0 = time.time()
        vecs = model.encode(texts, convert_to_numpy=True)
        ms = (time.time() - t0) * 1000
        norm = float(np.linalg.norm(vecs, axis=1).mean())
        print(f"shape={{vecs.shape}} norm={{norm:.3f}} ms={{ms:.0f}}")
    """)
    r = subprocess.run([py, "-c", verify_script], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        ok(f"推理验证通过  {out}")
    else:
        warn("推理验证失败 (非致命)")


# ── 6. 写 MCP 配置 ───────────────────────────────────────────────────
def write_mcp_config(venv: Path, model_key: str, gpu: bool) -> tuple[str, str]:
    step("生成 MCP 配置")

    py   = _python(venv)
    src  = str(SRC)
    db   = str(Path.home() / ".memorme" / "memory.db")
    use_gpu = "true" if gpu else "false"

    mcp_block = {
        "command": py,
        "args": ["-m", "memory"],
        "cwd": src,
        "env": {
            "MEMORY_DB_PATH": db,
            "MEMORY_USE_GPU": use_gpu,
            "MEMORY_MODEL": model_key,
        },
    }

    # ── Claude Desktop ───────────────────────────────────────────
    if sys.platform == "win32":
        claude_dir = Path(os.environ.get("APPDATA", "")) / "Claude"
    elif sys.platform == "darwin":
        claude_dir = Path.home() / "Library" / "Application Support" / "Claude"
    else:
        claude_dir = Path.home() / ".config" / "claude"

    claude_cfg = claude_dir / "claude_desktop_config.json"
    claude_dir.mkdir(parents=True, exist_ok=True)

    if claude_cfg.exists():
        try:
            cfg = json.loads(claude_cfg.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}

    cfg.setdefault("mcpServers", {})["memorme"] = mcp_block
    claude_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    ok(f"Claude Desktop → {claude_cfg}")

    # ── VS Code (.vscode/mcp.json) ───────────────────────────────
    vscode_dir = ROOT / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    vscode_mcp = vscode_dir / "mcp.json"

    vscode_cfg = {
        "servers": {
            "memorme": {
                "type": "stdio",
                "command": py,
                "args": ["-m", "memory"],
                "env": {
                    "PYTHONPATH": src,
                    "MEMORY_DB_PATH": db,
                    "MEMORY_USE_GPU": use_gpu,
                    "MEMORY_MODEL": model_key,
                },
            }
        }
    }
    vscode_mcp.write_text(json.dumps(vscode_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    ok(f"VS Code     → {vscode_mcp}")

    return str(claude_cfg), str(vscode_mcp)


# ── 7. 验证安装 ──────────────────────────────────────────────────────
def verify_install(venv: Path) -> None:
    step("验证安装")
    py = _python(venv)

    # tool 数量
    server_py = SRC / "memory" / "server.py"
    if server_py.exists():
        tree = ast.parse(server_py.read_text(encoding="utf-8"))
        tools = [
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and any(
                hasattr(d, "func") and getattr(d.func, "attr", "") == "tool"
                for d in n.decorator_list
            )
        ]
        ok(f"Server: {len(tools)} tools  {tools}")
    else:
        warn("server.py 未找到，跳过 tool 检查")

    # Late Chunking
    lc_script = textwrap.dedent(f"""\
        import sys; sys.path.insert(0, r"{SRC}")
        from memory.embedding import LATE_CHUNKING_MODELS
        print(LATE_CHUNKING_MODELS)
    """)
    r = subprocess.run([py, "-c", lc_script], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        ok(f"Late Chunking 模型: {r.stdout.strip()}")

    # GPU
    gpu_script = textwrap.dedent("""\
        import torch
        cuda = torch.cuda.is_available()
        mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"CUDA={cuda} MPS={mps}")
    """)
    r = subprocess.run([py, "-c", gpu_script], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        ok(f"GPU: {r.stdout.strip()}")


# ── 卸载 ─────────────────────────────────────────────────────────────
def uninstall(venv: Path) -> None:
    step("卸载 / 清理")
    if venv.exists():
        shutil.rmtree(venv)
        ok(f"已删除 {venv}")
    else:
        warn(".venv 不存在")
    ok("完成")


# ── 列出模型 ─────────────────────────────────────────────────────────
def list_models() -> None:
    header = f"{'Key':<15} {'大小':>6}  {'维度':>5}  {'MaxSeq':>6}  {'LateChunk':<10} 说明"
    print(bold(header))
    print("─" * 80)
    for key, m in MODELS.items():
        lc = _c(32, "✅") if m["late_chunking"] else dim("—")
        print(f"  {bold(key):<15} {m['size_mb']:>5}MB  {m['dim']:>5}  "
              f"{m['max_seq']:>6}  {lc:<10}  {dim(m['desc'])}")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MemorMe 一键部署脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python deploy.py                          # 默认: jina-v3, CPU
              python deploy.py --model bge-m3 --gpu     # bge-m3 + CUDA
              python deploy.py --download-only          # 只下载模型
              python deploy.py --list-models            # 查看可用模型
              python deploy.py --uninstall              # 删除 .venv
        """),
    )
    p.add_argument("--model", "-m", default="jina-v3",
                   choices=list(MODELS), metavar="MODEL",
                   help=f"嵌入模型 (默认: jina-v3)。可选: {', '.join(MODELS)}")
    p.add_argument("--gpu", "-g", action="store_true",
                   help="安装 CUDA 版 torch")
    p.add_argument("--cuda", default="cu121",
                   help="CUDA 版本标签 (默认: cu121)")
    p.add_argument("--skip-download", action="store_true",
                   help="跳过模型下载")
    p.add_argument("--skip-config", action="store_true",
                   help="跳过 MCP 配置写入")
    p.add_argument("--download-only", action="store_true",
                   help="仅下载模型 (不创建 venv / 不安装依赖)")
    p.add_argument("--list-models", action="store_true",
                   help="列出可用模型后退出")
    p.add_argument("--uninstall", action="store_true",
                   help="删除 .venv")
    p.add_argument("--no-verify", action="store_true",
                   help="跳过下载后推理验证")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    venv = ROOT / ".venv"

    print(bold("\n MemorMe 一键部署"))
    print(dim(f" 项目目录: {ROOT}\n"))

    # ── 特殊模式 ─────────────────────────────────────────────
    if args.list_models:
        list_models()
        return 0

    if args.uninstall:
        uninstall(venv)
        return 0

    if args.download_only:
        check_python()
        success = download_model(args.model, venv if venv.exists() else None,
                                 verify=not args.no_verify)
        return 0 if success else 1

    # ── 完整部署流程 ─────────────────────────────────────────
    check_python()
    create_venv(venv)
    upgrade_pip(venv)
    install_deps(venv, args.gpu, args.cuda)

    dl_ok = True
    if not args.skip_download:
        dl_ok = download_model(args.model, venv, verify=not args.no_verify)

    claude_cfg = vscode_cfg = ""
    if not args.skip_config:
        claude_cfg, vscode_cfg = write_mcp_config(venv, args.model, args.gpu)

    verify_install(venv)

    # ── 结果汇总 ─────────────────────────────────────────────
    db_path = str(Path.home() / ".memorme" / "memory.db")
    summary = [
        "MemorMe 部署完成！",
        f"模型   : {args.model}" + ("" if dl_ok else " (下载失败，请稍后重试)"),
        f"Python : {_python(venv)}",
        f"数据库 : {db_path}",
    ]
    if claude_cfg:
        summary.append(f"Claude : {claude_cfg}")
    if vscode_cfg:
        summary.append(f"VSCode : {vscode_cfg}")

    banner(summary)
    print("重启 Claude Desktop 或 VS Code 后即可使用 memorme。\n")
    return 0 if dl_ok else 1


if __name__ == "__main__":
    sys.exit(main())
