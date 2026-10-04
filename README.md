# Local Knowledge Base

一个从 `AgentDemo` 抽离出来的独立项目，提供：

- PDF、DOCX、Markdown、TXT 文档上传
- 文本解析、清洗、切块和常见个人信息脱敏
- 本地 Chroma 持久化向量库
- FastAPI Web 上传和检索界面
- Streamable HTTP MCP 服务

## 项目结构

```text
LocalKnowledgeBase/
├── knowledge_base/
│   ├── store.py       # 文档处理、Chroma 存储、安全检索出口
│   ├── sanitizer.py   # 入库和检索结果脱敏
│   ├── embeddings.py   # 本地模型/测试向量提供者
│   ├── web_app.py     # Web UI 和 REST API
│   └── mcp_server.py  # MCP tools/list、tools/call 服务
├── scripts/
│   └── migrate_agentdemo_data.py
├── tests/
├── data/              # Chroma 数据库、catalog.json、原始上传文件
└── pyproject.toml
```

运行测试：

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## 安装和运行

建议使用 Python 3.10+，并使用项目自己的虚拟环境：

```bash
cd /Users/wangyilei/Documents/opensource/LocalKnowledgeBase
python3.10 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

启动 Web 服务：

```bash
.venv/bin/python -m knowledge_base.web_app
```

默认打开 <http://127.0.0.1:8110>。

启动 MCP 服务：

```bash
MCP_HOST=127.0.0.1 MCP_PORT=8080 .venv/bin/python -m knowledge_base.mcp_server
```

MCP 地址为 `http://127.0.0.1:8080/mcp`，提供：

- `search_knowledge(query, n_results)`
- `list_knowledge()`
- `list_knowledge_files()`

## 从 AgentDemo 迁移已有上传文件

如果原项目和本项目位于同一个 `opensource` 目录下，可执行：

```bash
.venv/bin/python scripts/migrate_agentdemo_data.py
```

脚本会重新解析并写入新项目的 `data/`，不会复用原项目的 SQLite 文件，也不会把原项目的绝对路径带过来。

## 说明

当前保留确定性的本地字符/二元字符哈希向量，便于测试和离线演示；生产建议使用已经下载到本机的真实 Embedding 模型。项目只负责检索和脱敏，若要生成最终答案，可将安全的 `search_knowledge` 结果交给上层 LLM 形成完整 RAG 链路。

### 使用本地 Embedding 模型

安装可选依赖：

```bash
.venv/bin/pip install -e '.[local-embedding]'
```

将模型提前下载到本机，例如 `/models/bge-small-zh-v1.5`，然后配置：

```bash
export KNOWLEDGE_BASE_EMBEDDING_PROVIDER=local
export KNOWLEDGE_BASE_EMBEDDING_MODEL_PATH=/models/bge-small-zh-v1.5
export KNOWLEDGE_BASE_COLLECTION=knowledge_hub_bge_small_zh_v1
export KNOWLEDGE_BASE_SEED_DEMO=false
```

文档入库和查询必须使用同一个 Embedding 模型。更换模型或向量维度后，应使用新的 collection 并重新索引全部文档，不能直接混用旧向量。

切换模型后可以这样重建 collection：

```bash
KNOWLEDGE_BASE_COLLECTION=knowledge_hub_bge_small_zh_v1 \
KNOWLEDGE_BASE_EMBEDDING_PROVIDER=local \
KNOWLEDGE_BASE_EMBEDDING_MODEL_PATH=/models/bge-small-zh-v1.5 \
.venv/bin/python scripts/reindex.py
```

无模型的测试环境可以显式设置 `KNOWLEDGE_BASE_EMBEDDING_PROVIDER=hash`。

检索结果会在入库和输出两个阶段进行脱敏，外部结果只返回白名单 metadata，不返回数据库路径和原始文件路径。生产部署仍应增加访问认证、原始文件权限控制、备份恢复和监控；本项目没有保留原 `AgentDemo` 中的任何硬编码密钥。
