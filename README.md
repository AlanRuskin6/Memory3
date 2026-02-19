
# MemorMe

极致轻量的 AI 记忆系统，专为 [Model Context Protocol (MCP)](https://modelcontextprotocol.io) 设计。

语义记忆存储与混合搜索，完全本地运行，无需 Docker 或云服务。

---

## 特性

- **Late Chunking** — 先编码全文 token，再按边界切分做 mean pooling，每个 chunk embedding 携带全文上下文；比传统分块检索精度显著更高（ICLR 2025）
- **上下文窗口扩展** — `memory_search` 的 `context_window` 参数，命中 chunk_005 时自动拉取前后 N 块合并，LLM 一次搜索即获完整上下文
- **混合搜索 (RRF)** — 向量语义 + BM25 关键词融合排名，比纯向量搜索精度提升 15–30%
- **HNSW 索引** — O(log n) 近似最近邻搜索，10 万条记忆搜索延迟 < 10ms
- **智能文件导入** — 20+ 编程语言语法感知分块（Python/C++/Rust/Go 等），文本按段落/句子分割
- **标签关系表** — 独立 `memory_tags` 索引表，O(1) 标签查询，替代 JSON `LIKE` 扫描
- **Embedding 缓存** — 相同内容零重复编码，批量导入自动去重
- **TTL 支持** — 记忆可设置过期时间，自动清理
- **线程安全** — 信号量限制连接池 + WAL + RLock，支持多线程并发
- **分层降级** — HNSW → sqlite-vec → 纯 Python，缺少依赖时自动降级

---

## 安装

```bash
# 基础安装
pip install -e .

# 含开发依赖（测试）
pip install -e ".[dev]"

# 含 HNSW 索引支持
pip install -e ".[hnsw]"
```

**依赖要求**

| 包 | 版本 | 说明 |
|----|------|------|
| Python | >= 3.10 | |
| mcp | >= 1.1.0 | MCP 协议 |
| sentence-transformers | >= 3.0.0 | 嵌入模型 |
| transformers | >= 4.40.0 | Late Chunking 所需 token-level API |
| torch | >= 2.0.0 | 模型推理 |
| sqlite-vec | >= 0.1.0 | SQLite 向量扩展（可选，自动降级）|
| hnswlib | >= 0.8.0 | HNSW 索引（可选，自动降级）|
| numpy | >= 1.24.0 | 数值计算 |

---

## 快速开始

### MCP 配置

在 Claude Desktop 或其他 MCP 客户端的配置文件中添加：

```json
{
  "mcpServers": {
    "memorme": {
      "command": "python",
      "args": ["-m", "memory"],
      "env": {
        "MEMORY_DB_PATH": "~/.memorme/memory.db",
        "MEMORY_USE_GPU": "true",
        "MEMORY_MODEL": "jina-v3"
      }
    }
  }
}
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEMORY_DB_PATH` | `~/.memorme/memory.db` | 数据库文件路径 |
| `MEMORY_USE_GPU` | `true` | 是否启用 GPU |
| `MEMORY_MODEL` | `bge-m3` | 嵌入模型预设（见下表）|

### 可用模型

| Key | 模型 | 大小 | 最大 Token | Late Chunking | 说明 |
|-----|------|------|-----------|--------------|------|
| `jina-v3` | jinaai/jina-embeddings-v3 | 570MB | **8192** | ✅ | **推荐**，长上下文，Late Chunking 原生支持 |
| `bge-m3` | BAAI/bge-m3 | 2.2GB | 8192 | — | 最高质量，100+ 语言 |
| `bge-small-zh` | BAAI/bge-small-zh-v1.5 | 95MB | 512 | — | 轻量中文 |
| `bge-small-en` | BAAI/bge-small-en-v1.5 | 130MB | 512 | — | 轻量英文 |
| `minilm` | all-MiniLM-L6-v2 | 90MB | 256 | — | 超轻量回退 |

> **Late Chunking** 需要长上下文模型（8192+ tokens），目前已配置 `jina-v3`。
> 使用其他模型时，文件导入自动回退到标准批量编码。

---

## 一键部署

`deploy.py` 完成「环境检查 → 虚拟环境 → 依赖安装 → 模型下载 → MCP 配置写入 → 验证」全流程，跨平台 (Windows / macOS / Linux)。

```bash
# 默认部署 (jina-v3, CPU)
python deploy.py

# 指定模型
python deploy.py --model bge-m3

# CUDA 加速
python deploy.py --gpu
python deploy.py --gpu --cuda cu121    # 指定 CUDA 版本

# 仅下载模型 (不建 venv)
python deploy.py --download-only

# 跳过模型下载 / 跳过配置写入
python deploy.py --skip-download
python deploy.py --skip-config

# 列出可用模型
python deploy.py --list-models

# 卸载 (删除 .venv)
python deploy.py --uninstall
```

脚本完成后会自动写入：
- **Claude Desktop** 配置（Windows `%APPDATA%\Claude\`；macOS `~/Library/Application Support/Claude/`）
- **VS Code** 配置（`.vscode/mcp.json`）

重启 Claude Desktop 或 VS Code 后即可使用。

---

## 完整部署指南

### 第一步：环境准备

```powershell
# Windows PowerShell
python --version   # 需要 >= 3.10

# 建议使用虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 安装所有依赖
pip install -e ".[hnsw]"
pip install "transformers>=4.40.0" torch
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[hnsw]"
pip install "transformers>=4.40.0" torch
```

### 第二步：选择并下载模型

**方案 A：jina-v3（推荐，支持 Late Chunking）**

```bash
# 首次运行时自动下载，约 570MB
# 预先下载（可选）：
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)"
```

MCP 配置中设置：`"MEMORY_MODEL": "jina-v3"`

**方案 B：bge-m3（最高精度，无 Late Chunking）**

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

MCP 配置中设置：`"MEMORY_MODEL": "bge-m3"`

**方案 C：轻量级（内存有限 / 低配机器）**

```json
"MEMORY_MODEL": "bge-small-zh"   // 95MB，中文场景
"MEMORY_MODEL": "minilm"          // 90MB，英文场景
```

### 第三步：配置 MCP 客户端

**Claude Desktop**（Windows：`%APPDATA%\Claude\claude_desktop_config.json`）

```json
{
  "mcpServers": {
    "memorme": {
      "command": "C:\\Users\\你的用户名\\.venv\\Scripts\\python.exe",
      "args": ["-m", "memory"],
      "cwd": "C:\\Users\\你的用户名\\Desktop\\ProJect\\Other\\memory3\\src",
      "env": {
        "MEMORY_DB_PATH": "C:\\Users\\你的用户名\\.memorme\\memory.db",
        "MEMORY_USE_GPU": "true",
        "MEMORY_MODEL": "jina-v3"
      }
    }
  }
}
```

**VS Code（mcp.json）**

```json
{
  "servers": {
    "memorme": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "memory"],
      "env": {
        "PYTHONPATH": "C:\\Users\\你的用户名\\Desktop\\ProJect\\Other\\memory3\\src",
        "MEMORY_DB_PATH": "C:\\Users\\你的用户名\\.memorme\\memory.db",
        "MEMORY_USE_GPU": "true",
        "MEMORY_MODEL": "jina-v3"
      }
    }
  }
}
```

### 第四步：GPU 加速（可选）

```bash
# NVIDIA GPU（CUDA）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon（MPS，自动检测）
pip install torch   # 已内置 MPS 支持

# 确认设备
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
```

### 第五步：验证安装

```powershell
# 确认 9 个 tool 注册成功
python -c "
import ast, pathlib
tree = ast.parse(pathlib.Path('src/memory/server.py').read_text(encoding='utf-8'))
tools = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
         and any(hasattr(d, 'func') and getattr(d.func, 'attr', '') == 'tool'
                 for d in n.decorator_list)]
print(f'{len(tools)} tools:', tools)
"
# 预期输出: 9 tools: ['memory_save', 'memory_get', ...]

# 确认 Late Chunking 支持
python -c "
import os; os.chdir('src')
from memory.embedding import EmbeddingEngine, LATE_CHUNKING_MODELS
print('Late Chunking 支持的模型:', LATE_CHUNKING_MODELS)
"
# 预期输出: Late Chunking 支持的模型: {'jina-v3'}
```

### 更换模型后重新导入

> **重要**：不同模型生成的 embedding 维度和空间不同，无法混用。
> 更换模型后必须重新导入所有文件，旧 embedding 自动失效。

```python
# 1. 清理旧的文件导入记忆
memory_delete(by="prefix", prefix="file_")

# 2. 用新模型重新导入
memory_import(action="file", file_path="/path/to/your/file.py", tags=["codebase"])
```

---

## MCP 工具参考

共 **9 个工具**，每个工具通过参数路由覆盖多种操作模式。

### 1. `memory_save` — 保存记忆

单条或批量保存。传 `items` 参数时自动走批量模式（去重编码）。

```python
# 单条保存
memory_save(
    content="Q4 规划会议：确定了三条产品路线",
    key="meeting_2024_q4",          # 可选，不填自动生成
    tags=["work", "meeting"],
    metadata={"project": "alpha"},
    ttl_seconds=604800              # 7天后过期，可选
)

# 批量保存
memory_save(items=[
    {"content": "记忆1", "tags": ["tag1"]},
    {"content": "记忆2", "metadata": {"type": "note"}},
])
```

### 2. `memory_get` — 按 key 获取

```python
memory_get(key="meeting_2024_q4")
```

### 3. `memory_search` — 搜索记忆

通过 `mode` 参数选择搜索策略：`hybrid`（默认）、`vector`、`text`。
`context_window` 参数自动拉取相邻 chunk，LLM 一次搜索即获完整上下文。

```python
# 混合搜索（推荐，默认）
memory_search(
    query="Q4 产品规划",
    top_k=10,
    threshold=0.3,
    tags_filter=["work"],
    metadata_filter={"project": "alpha"},
    vector_weight=0.5,
    text_weight=0.5
)

# 纯语义搜索
memory_search(query="产品规划", mode="vector")

# 纯关键词搜索（FTS5，无需 embedding）
memory_search(query="Q4 规划", mode="text")

# 上下文窗口扩展（推荐用于代码/长文档）
# 命中 chunk_005 时自动合并 chunk_004 + chunk_005 + chunk_006 返回
memory_search(query="初始化逻辑", context_window=1)

# 更大窗口（命中块前后各2块）
memory_search(query="错误处理", context_window=2)
```

**`context_window` 说明：**
- 仅对通过 `memory_import(action="file")` 导入的分块记忆生效
- 非分块记忆（无 `chunk_index` metadata）原样返回
- 返回结果中包含 `context_expanded: true` 和 `context_chunks: [start, end]` 字段

### 4. `memory_delete` — 删除记忆

通过 `by` 参数选择删除策略：`key`（默认）、`expired`、`tag`、`prefix`。

```python
memory_delete(key="meeting_2024_q4")                  # 按 key
memory_delete(by="expired")                            # 清理所有过期记忆
memory_delete(by="tag", tag="temp")                    # 按标签
memory_delete(by="prefix", prefix="file_report_")     # 按 key 前缀
```

### 5. `memory_update` — 更新元数据和/或标签

`metadata` 和 `tags` 至少提供一个。`merge=True`（默认）合并，`merge=False` 替换。

```python
memory_update(
    key="meeting_2024_q4",
    metadata={"priority": "high"},
    tags=["important"],
    merge=True
)
```

### 6. `memory_list` — 列表查询

`tags` 参数非空时走标签过滤。

```python
memory_list(limit=20, offset=0)                    # 全量分页
memory_list(tags=["work", "meeting"], limit=50)    # 按标签筛选
```

### 7. `memory_import` — 导入/导出

通过 `action` 参数选择：`json`（默认）、`export`、`file`。

```python
# 导出全部记忆
result = memory_import(action="export")
# result["memories"] 包含所有记忆数据

# 从导出数据导入
memory_import(action="json", memories=result["memories"])

# 从文件导入（智能分块：代码按函数/类边界，文本按段落）
memory_import(
    action="file",
    file_path="/path/to/main.py",
    chunk_size=2000,
    overlap=200,
    min_chunk_size=50,
    tags=["codebase"],
    metadata={"repo": "my_project"}
)
```

支持的代码格式：`.py` `.cpp` `.c` `.h` `.js` `.ts` `.rs` `.go` `.java` `.cs` `.lua` `.rb` `.php` `.swift` `.kt` `.scala` `.zig` 等。

> **Late Chunking 自动启用**：使用 `jina-v3` 模型时，文件导入自动使用 Late Chunking
> 编码——全文过一次 Transformer，再按边界 mean pool，每个 chunk 携带全文上下文。
> 切换到 `bge-m3` 等短上下文模型时，自动回退到标准批量编码。

### 8. `memory_stats` — 统计与清理

默认返回系统统计。设置 `cleanup=True` 触发垃圾片段清理。

```python
memory_stats()
# 返回: total_memories, active, expired, embedding_model, device...

memory_stats(cleanup=True, cleanup_min_length=50, cleanup_dry_run=True)
# 清理内容过短的垃圾片段，dry_run=True 时仅预览不删除
```

### 9. `memory_similarity` — 文本相似度

```python
memory_similarity(text1="今天天气好", text2="阳光明媚")
# 返回: {"similarity": 0.85, "interpretation": "Very similar"}
```

---

## Python API

可直接在 Python 脚本中使用，无需 MCP：

```python
from memory.memory import MemoryManager

manager = MemoryManager(
    db_path="./memory.db",
    model_key="bge-m3",
    use_gpu=True
)

# 保存
key = manager.save(
    content="Python 是一门优雅的语言",
    tags=["programming"],
    metadata={"source": "notes"}
)

# 混合搜索
results = manager.search_hybrid(query="编程语言", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.content[:80]}")

# 批量保存（自动去重编码）
keys = manager.save_batch([
    {"content": "记忆A", "tags": ["a"]},
    {"content": "记忆B"},
])

manager.close()
```

---

## 架构

```
MCP Server (server.py)         9 个工具定义
       |
Memory Manager (memory.py)     高层 API + Embedding 缓存 + Late Chunking
       |
 ┌─────┴─────┐
 |           |
Embedding   Vector Storage (storage.py)
(embedding.py)   SQLite + WAL + FTS5 + Tags 关系表
     |               |
sentence-        ┌───┴───┐
transformers   HNSW   sqlite-vec
(+ Late Chunking  (hnsw_index.py)
 for jina-v3)
```

**Late Chunking 流程（jina-v3 模型）：**
```
全文 → Transformer → token embeddings (8192 tokens)
                          ↓
每个 chunk 边界做 mean pooling
                          ↓
chunk embedding 携带全文上下文
```

**Context Window Expansion 流程（查询时）：**
```
搜索命中 chunk_005
    ↓ context_window=1
自动 GET chunk_004 + chunk_006
    ↓
合并为完整上下文返回给 LLM
```

**搜索优先级**：HNSW (O(log n)) → sqlite-vec (SIMD KNN) → 纯 Python 余弦相似度

---

## 性能参考

| 操作 | CPU | GPU |
|------|-----|-----|
| 单条 encode | ~50ms | ~10ms |
| 批量 encode ×100 | ~2s | ~200ms |
| 向量搜索（10k 条，HNSW）| < 10ms | < 10ms |
| 全文搜索（FTS5）| < 5ms | — |
| 混合搜索 | < 20ms | < 15ms |
| 标签过滤查询 | < 1ms | — |
| serialize_f32 (384d) | ~2μs | — |

---

## 项目结构

```
ultra-light-memory/
├── src/memory/
│   ├── __init__.py
│   ├── __main__.py        入口点
│   ├── server.py          MCP 工具定义（9 个工具）
│   ├── memory.py          高层记忆管理 API + Embedding 缓存
│   ├── embedding.py       嵌入引擎，多模型支持
│   ├── storage.py         向量存储 + 混合搜索 + Tags 关系表
│   └── hnsw_index.py      HNSW 近似最近邻索引
├── tests/
│   └── test_memory.py     36 项测试
├── cleanup_db.py          清理垃圾片段的独立脚本
└── pyproject.toml
```

---

## 测试

```bash
# 设置 PYTHONPATH 指向 src 目录
$env:PYTHONPATH = "src"   # Windows PowerShell
export PYTHONPATH=src      # Linux/macOS

pytest tests/ -v
pytest tests/ --cov=memory --cov-report=html
```

---

## v0.6.0 更新日志

### 一键部署脚本 `deploy.py`
- **跨平台 Python 脚本**，替代原 `setup.ps1` / `setup.sh`，Windows / macOS / Linux 同一入口
- **七步自动化流程**：Python 版本检查 → 创建 `.venv` → 升级 pip → 安装 `.[hnsw]` + `transformers` / `torch` → 模型下载 → 写入 MCP 配置 → 安装验证
- **自动写入 MCP 配置**
  - Claude Desktop：Windows `%APPDATA%\Claude\claude_desktop_config.json`；macOS `~/Library/Application Support/Claude/`；Linux `~/.config/claude/`
  - VS Code：`.vscode/mcp.json`（`PYTHONPATH` 自动指向 `src/`）
- **模型下载**：下载时显示旋转进度符；完成后自动运行推理验证（shape / norm / 耗时），可用 `--no-verify` 跳过
- **命令行参数**

  | 参数 | 说明 |
  |------|------|
  | `--model MODEL` | 选择嵌入模型（默认 `jina-v3`）|
  | `--gpu` | 安装 CUDA 版 torch |
  | `--cuda VER` | 指定 CUDA 标签（默认 `cu121`）|
  | `--skip-download` | 跳过模型下载 |
  | `--skip-config` | 跳过 MCP 配置写入 |
  | `--download-only` | 仅下载模型，不建虚拟环境 |
  | `--list-models` | 表格列出 5 个可用模型 |
  | `--no-verify` | 跳过推理验证 |
  | `--uninstall` | 删除 `.venv` |

- **验证输出**：检查 server.py 中注册的 tool 数量、Late Chunking 支持的模型集合、CUDA/MPS 可用状态
- **GBK 终端乱码修复**：Windows 启动时自动 `chcp 65001` + `stdout.reconfigure(encoding='utf-8')`，彻底解决中文显示问题

---

## v0.5.0 更新日志

### Late Chunking（ICLR 2025 技术）
- **新增 `jina-v3` 模型** — jinaai/jina-embeddings-v3，570MB，8192 token 上下文，原生支持 Late Chunking
- **`EmbeddingEngine.encode_late_chunks()`** — 全文单次 Transformer 编码，按 chunk 边界 mean pool token embeddings，每个 chunk embedding 携带全文语义上下文
- **自动启用** — `memory_import(action="file")` 检测模型是否支持，自动选择 Late Chunking 或标准编码
- **`MemoryManager.save_batch_late_chunking()`** — 新增高层 API，传入全文和 chunk 列表，自动处理 Late Chunking
- **自动降级** — 非 Late Chunking 模型、文档超 8192 token，或 token 边界定位失败时，平滑降级到标准 encode

### Context Window Expansion
- **`memory_search` 新增 `context_window` 参数**（0-10，默认 0）——命中分块记忆时，自动拉取前后 N 个相邻 chunk 并合并内容返回
- 基于 metadata `chunk_index` / `total_chunks` 推断邻块 key，无需额外存储
- 返回结果增加 `context_expanded: true` 和 `context_chunks: [start_idx, end_idx]` 字段
- 内置去重：同一文件多个 chunk 命中时不重复拉取邻块

---

## v0.4.0 更新日志

### 工具合并（19 → 9）
- **memory_save** — 合并 `save` + `save_batch`，`items` 参数非空时走批量
- **memory_search** — 合并 `search` + `search_text`，`mode` 参数：`hybrid` / `vector` / `text`
- **memory_delete** — 合并 `delete` + `delete_expired` + `delete_by_tag` + `delete_by_prefix`，`by` 参数路由
- **memory_update** — 合并 `update_metadata` + `update_tags`，可同时更新
- **memory_list** — 合并 `list` + `list_by_tags`，`tags` 参数非空时走标签查询
- **memory_import** — 合并 `import` + `import_file` + `export`，`action` 参数路由
- **memory_stats** — 合并 `stats` + `cleanup`，`cleanup` 参数触发清理
- 后端层（storage.py, memory.py）不变，现有测试无需修改

---

## v0.3.0 更新日志

### 性能优化
- **serialize_f32** — `struct.pack` → `ndarray.tobytes()`，序列化速度提升 10–50×
- **deserialize_f32** — `struct.unpack` → `np.frombuffer`，零拷贝反序列化
- **Embedding 缓存** — LRU 缓存（最大 10k 条），相同内容零重复编码
- **批量去重编码** — `save_batch` 自动检测重复内容，只编码唯一文本
- **延迟 Embedding 解析** — `_parse_row` 默认不反序列化 embedding blob，列表/搜索路径内存占用大幅降低
- **显式列选择** — 查询不再 `SELECT *`，避免加载 embedding 列

### 数据层改进
- **Tags 关系表** — 新增 `memory_tags` 索引表，`list_by_tags` 从全表扫描 → `JOIN + INDEX`
- **自动迁移** — 首次启动自动从 JSON `tags` 列填充关系表，无需手动操作
- **连接池信号量** — 限制最大并发连接数，防止无限创建连接
- **HNSW 删除同步** — `delete` / `delete_expired` 同步清理 HNSW 索引，防止幽灵结果
- **datetime 索引修复** — 5 处 `datetime(expires_at)` → 直接字符串比较，索引可用

### 正确性修复
- **similarity 双重归一化** — 移除冗余 norm 计算，`encode(normalize=True)` 已保证单位向量
- **包名一致性** — 项目重命名为 `MemorMe`，`pyproject.toml` 入口点与实际包目录对齐
- **测试导入修复** — 38 处 `ultra_light_memory` → `memory`

---

## License

MIT
=======
# Memory3
极致轻量的 AI 记忆MCP系统
>>>>>>> f5c8a55fd00243b5e61d020f059c6ab067f63386
