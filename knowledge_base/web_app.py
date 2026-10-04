"""FastAPI web UI for uploading and searching local knowledge."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .store import build_state, get_file_path, ingest_file, search_documents


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Knowledge Base</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#182230;background:#f5f7fa}
section{background:#fff;border:1px solid #dbe3eb;border-radius:12px;padding:20px;margin:16px 0}h1{margin-top:0}
button{background:#146c94;color:#fff;border:0;border-radius:6px;padding:9px 14px;cursor:pointer}input{padding:9px;border:1px solid #c7d2dc;border-radius:6px}pre{white-space:pre-wrap;background:#f7f9fb;padding:12px;border-radius:6px}
.muted{color:#617184}.result{border-top:1px solid #e6ebf0;padding:12px 0}.result:first-child{border-top:0}
</style></head><body><h1>Local Knowledge Base</h1>
<p class="muted">上传文档，写入 Chroma 后进行本地语义检索。原始文件保存在项目 data/uploads。</p>
<section><h2>状态</h2><pre id="state">加载中...</pre></section>
<section><h2>上传文档</h2><form id="upload"><input type="file" name="files" multiple accept=".pdf,.docx,.md,.markdown,.txt"> <button>上传</button></form><p id="uploadMsg"></p></section>
<section><h2>检索</h2><form id="search"><input id="query" size="50" placeholder="例如：Chroma 是什么" required> <button>搜索</button></form><div id="results"></div></section>
<script>
const state=document.querySelector('#state'),msg=document.querySelector('#uploadMsg'),results=document.querySelector('#results');
async function refresh(){const r=await fetch('/api/state');state.textContent=JSON.stringify(await r.json(),null,2)}
document.querySelector('#upload').onsubmit=async e=>{e.preventDefault();msg.textContent='上传中...';const r=await fetch('/api/upload',{method:'POST',body:new FormData(e.target)});const d=await r.json();msg.textContent=r.ok?`已处理 ${d.inserted_chunks} 个文档块`:d.detail;await refresh()};
document.querySelector('#search').onsubmit=async e=>{e.preventDefault();results.textContent='搜索中...';const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:document.querySelector('#query').value,top_k:5})});const d=await r.json();results.replaceChildren();(d.results||[]).forEach(x=>{const el=document.createElement('div');el.className='result';const title=document.createElement('strong');title.textContent=(x.file_name||'demo')+(x.page_number?' · 第'+x.page_number+'页':'');const body=document.createElement('div');body.textContent=x.snippet||'';const score=document.createElement('small');score.textContent=`distance: ${Number(x.distance||0).toFixed(4)}`;el.append(title,body,score);results.appendChild(el)})};refresh();
</script></body></html>"""


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


app = FastAPI(title="Local Knowledge Base")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
def api_state():
    return build_state()


@app.post("/api/search")
def api_search(payload: SearchRequest):
    return {"query": payload.query, "results": search_documents(payload.query, payload.top_k)}


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Please select at least one file.")
    items, inserted_chunks = [], 0
    for file in files:
        max_bytes = int(os.getenv("KNOWLEDGE_BASE_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            items.append({"file_name": file.filename or "upload.bin", "status": "error", "message": "File is too large."})
            continue
        if not data:
            continue
        try:
            item = ingest_file(file.filename or "upload.bin", file.content_type, data)
            inserted_chunks += item["chunk_count"]
            items.append({**item, "status": "ok"})
        except Exception as exc:
            items.append({"file_name": file.filename or "upload.bin", "status": "error", "message": str(exc)})
    if not items:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    return {"items": items, "inserted_chunks": inserted_chunks}


@app.get("/api/files/{file_hash}")
def download_file(file_hash: str):
    if os.getenv("KNOWLEDGE_BASE_ENABLE_RAW_DOWNLOAD", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=404, detail="File download is disabled.")
    item = get_file_path(file_hash)
    if item is None:
        raise HTTPException(status_code=404, detail="File not found.")
    path, filename = item
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored file not found.")
    return FileResponse(path, filename=filename)


def main() -> None:
    host = os.getenv("CHROMA_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("CHROMA_WEB_PORT", "8110"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
