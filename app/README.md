# INSARAG 东盟多语种搜救智能平台

FastAPI + SQLite + 静态前端，双嵌入后端（智谱 BigModel / 本地 sentence-transformers）+ DeepSeek 流式大模型。
覆盖：多源信息聚合、RAG 智能问答、东盟 10 语互译（术语强制译法）、语音翻译、知识库 CMS。

## 目录结构

```
app/
├── main.py                  # FastAPI 入口（非阻塞启动：冷启动只加载已建索引）
├── api/index.py             # Vercel serverless 入口（暴露 app）
├── netlify/functions/app.py # Netlify Functions 入口
├── vercel.json / netlify.toml
├── build_static.sh          # 静态前端独立打包（out/）
├── core/
│   ├── config.py           # 路径/环境变量/语种表；只读 FS 自动把 app.db 复制到 /tmp
│   ├── db.py               # SQLite schema、迁移、CRUD；可选 Turso/libSQL
│   ├── embedding.py        # 嵌入适配层：bigmodel / local / auto，统一 encode() 接口
│   ├── retrieval.py        # numpy 余弦 Top-K 内存索引
│   ├── llm.py               # DeepSeek OpenAI 兼容流式客户端
│   └── ingest.py           # 增量入库：跳过已切片、逐行提交、进度上报
├── routers/                 # FastAPI 路由（cms / rag / translate / ingest）
├── scripts/build_index.py   # 离线建库脚本（服务外运行，可中断续跑）
├── static/                  # 门户 / 聚合 / CMS / RAG / 翻译 / 语音 六个页面
└── data/app.db              # 预构建 SQLite（含向量，随部署包发布）
```

## 本地启动

```bash
cd app
pip install -r requirements.txt

# 配置密钥（权限 600，不入库）；模板见 .env.example
#   DEEPSEEK_API_KEY=...
#   EMBEDDING_BACKEND=bigmodel
#   BIGMODEL_API_KEY=...
#   BIGMODEL_EMBEDDING_DIM=512

# 首次（或数据更新后）离线建库：增量、可 Ctrl-C 续跑
python scripts/build_index.py
# 维度切换/清库重建：python scripts/build_index.py --wipe

# 启动服务（冷启动立即就绪；索引为空时才后台异步构建）
uvicorn main:app --host 0.0.0.0 --port 8000
```

浏览器打开 <http://127.0.0.1:8000/>。

## 嵌入后端

| 后端 | 说明 |
|---|---|
| `bigmodel` | 智谱 `POST {BIGMODEL_BASE_URL}/embeddings`，Key 从 `BIGMODEL_API_KEY` 读，Bearer 鉴权，模型 `BIGMODEL_EMBED_MODEL`（embedding-3），`BIGMODEL_EMBEDDING_DIM=512` 降维 |
| `local` | 本地 sentence-transformers；模型不可用时降级 TF-IDF，余弦仍由 numpy 真实计算 |
| `auto`（默认） | 有 BigModel Key 用 bigmodel，否则 local；调用失败自动 fallback 并在状态接口标注实际后端 |

**维度变更必须干净重建**（`build_index.py --wipe`），bigmodel(512) 与 local(384) 不可混存。

## 离线建库脚本（scripts/build_index.py）

- 在 Web 服务**之外**运行完整增量构建，打印进度（done/total、当前阶段、后端、错误）。
- 每条切片按 `(source_type, source_id, chunk_index)` 幂等：已切片的行跳过，不重复计费；
  逐行提交，Ctrl-C 后重跑自动续建。
- 自动检测已存向量维度 ≠ 当前后端维度时清空重建。
- 结束打印：行数、向量总数、维度、后端、探针余弦复算结果。

## Serverless 部署

冷启动**不重新嵌入**（数千次 BigModel 调用必然超时）。正确姿势：

1. **本地离线建好 `data/app.db`**（含全部向量，~27MB），随函数包一起发布。
2. 函数冷启动只读加载；只读 FS 上 `core/config.py` 自动把 `app.db` 复制到 `/tmp/app.db`。
3. 环境变量在平台控制台配置（不写进代码）。

### Vercel

```bash
cd app
# 环境变量：DEEPSEEK_API_KEY / BIGMODEL_API_KEY / EMBEDDING_BACKEND=bigmodel
vercel --prod
```

`vercel.json` 把所有路径 rewrite 到 `api/index.py`（ASGI handler），由 main.py 同时挂载 API 与静态页。

### Netlify

`netlify.toml`：函数 `netlify/functions/app.py`（ASGI），静态 `publish=out/`，全部流量 200 回函数。

### 静态前端独立托管（Cloudflare Pages / Vercel Static）

```bash
bash build_static.sh   # -> out/
```

把 `out/` 部署到 Pages，在页面里注入 `window.API_BASE = "https://你的api域名"`（js/app.js 会读取）。

### 可选：Turso / libSQL

设置 `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN`（并 `pip install libsql`），读写自动走远程库；
未安装/失败时自动回退本地 SQLite，不阻塞启动。

## API 要点

- 计数：news 10 / aggregator 103（AHA Centre 45 + UNDRR 14 + GDACS 44）/ corpus 2198 平行句对 / terminology 558 条；向量 2982 切片 × 512 维。
- `GET /api/status`：LLM/嵌入配置、计数、`build` 构建进度（status/done/total/backend/error）。
- `POST /api/rag/ask`：SSE；`source_types` 过滤四库（news/aggregator/corpus/term）；
  最高相关度 < `RAG_MIN_SCORE`（默认 0.5，实测无关题 0.36–0.48、相关题 0.61–0.83）时返回 `refuse` 事件并直接结束，不调用 LLM。
- `POST /api/translate/stream`：SSE；命中术语库强制规范译法，返回 `terms` 事件用于前后端高亮；附 RAG 来源。
- `POST /api/cms/review/{table}/{id}`：审核状态切换（已审核/待审核）。
- `POST /api/ingest/build-async`：后台异步增量重建（不阻塞请求）。

## 自测清单

| 检查 | 命令/动作 |
|---|---|
| 后端联通 | `python scripts/build_index.py` 结尾 probe 余弦复算 ok |
| 非阻塞启动 | `uvicorn main:app` 后立即 `GET /api/status`，`embedding.rows` 即索引行数 |
| 幂等续建 | 再跑一次 build_index.py，`added=0 / skipped=全部` |
| RAG 拒答 | 问一个与领域无关的问题，出现 refuse 横幅且不调模型 |
| 术语高亮 | 翻译含 OSOCC/USAR 的句子，原文/译文出现 mark 高亮 |
| 来源可点 | 来源区显示 URL（新窗口）+ 页码 badge |

## 截图

真实整页 PNG 见项目根 `screenshots/`：`portal/`、`aggregator/`、`cms/`（含 `*-viewport.png` 视口高度版，避免长表整页过长无法查看）、`rag/`（含 `rag-hard-refuse-threshold-0.5.png` 硬拒答横幅）、`translation/`、`voice/`。
