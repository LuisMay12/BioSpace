# app.py
import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from sentence_transformers import SentenceTransformer
from urllib.parse import quote_plus
from datetime import datetime

# (Opcional) LLM para generación
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ===== Configuración general =====
DB_NAME = os.getenv("DB_NAME", "BioSearch")
COLL = os.getenv("COLL", "papers_v2")
ATLAS_VECTOR_INDEX = os.getenv("ATLAS_VECTOR_INDEX", "default2")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ===== Credenciales Mongo Atlas =====
USER = os.getenv("MONGO_USER", "luismayrdz_db_user")
PASSWORD = os.getenv("MONGO_PASS", "tNXwmDPBL4hwyhpu") 
DB = DB_NAME
REPLICA_SET = "atlas-x2ji3e-shard-0"  # de tu DNS TXT

PWD = quote_plus(PASSWORD)
SEEDS = (
    "ac-mzuauq7-shard-00-00.sgh5mic.mongodb.net:27017,"
    "ac-mzuauq7-shard-00-01.sgh5mic.mongodb.net:27017,"
    "ac-mzuauq7-shard-00-02.sgh5mic.mongodb.net:27017"
)

# ===== URI completa (sin SRV, funcional en Windows) =====
MONGO_URI = (
    f"mongodb://{USER}:{PWD}@{SEEDS}/{DB}"
    f"?tls=true&replicaSet={REPLICA_SET}&authSource=admin"
    f"&retryWrites=true&w=majority&appName=BioSpacedb"
)

# ===== Conexión =====
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15000)
    client.admin.command("ping")
    print("✅ Conectado correctamente a MongoDB Atlas.")
except Exception as e:
    print("❌ Error al conectar con MongoDB Atlas:", e)
    raise

emb_model = SentenceTransformer(EMB_MODEL)
col = client[DB_NAME][COLL]
app = FastAPI(title="RAG Searcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cambiar a dominios específicos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGQuery(BaseModel):
    query: str
    k: int = 6
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    autores: Optional[List[str]] = None
    categorias: Optional[List[str]] = None
    tipo_articulo: Optional[str] = None
    generate: bool = False
    max_context_chars: int = 3000

def _build_filter(q: RAGQuery):
    match = {}
    if q.year_from or q.year_to:
        match["year"] = {}
        if q.year_from: match["year"]["$gte"] = q.year_from
        if q.year_to: match["year"]["$lte"] = q.year_to
    if q.tipo_articulo:
        match["tipo_articulo"] = q.tipo_articulo
    if q.autores:
        match["autores"] = {"$in": q.autores}
    if q.categorias:
        match["categorias"] = {"$in": q.categorias}
    return match

def _format_citation(doc):
    title = doc.get("titulo") or ""
    pid = doc.get("_id")
    year = doc.get("publication_date")
    doi = doc.get("doi")
    url = doc.get("url")
    return f"[{pid} - {year}] {title} (DOI: {doi}) {url or ''}".strip()

def _compose_context(docs: List[dict], max_chars: int):

    parts = []
    total = 0
    for d in docs:
        head = f"### {_format_citation(d)}\n"
        body = (d.get("abstract") or "")[:800]
        block = head + body + "\n\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)

def _generate_answer(query: str, context: str):
    prompt = f"""Eres un asistente que responde SOLO con información proveniente del contexto y agrega citas al final de cada afirmación con el formato [PMCID - año].
Si la información no está en el contexto, admite la limitación.

Pregunta:
{query}

Contexto:
{context}

Responde en español, conciso, estructurado en viñetas cuando sea útil.
"""
    if not USE_OPENAI:
        return "Contexto relevante:\n" + context[:1200] + ("\n… (truncado)" if len(context) > 1200 else "")

    # Con OpenAI (ejemplo con responses)
    resp = openai_client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=prompt,
        temperature=0.2,
    )
    return resp.output_text

@app.post("/rag")
def rag(q: RAGQuery):
    q_vec = emb_model.encode(q.query).tolist()
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_INDEX,
                "path": "embedding",
                "queryVector": q_vec,
                "numCandidates": max(50, q.k * 8),
                "limit": q.k
            }
        },
        {"$project": {
            "titulo": 1, "abstract": 1, "url": 1, "publication_date": 1, "doi": 1,
            "autores": 1, "categorias": 1, "tipo_articulo": 1,
            "score": {"$meta": "vectorSearchScore"}
        }},
    ]

    match = _build_filter(q)
    if match:
        pipeline.append({"$match": match})

    results = list(col.aggregate(pipeline))

    context = _compose_context(results, q.max_context_chars)
    answer = _generate_answer(q.query, context) if q.generate else None

    return {
        "query": q.query,
        "count": len(results),
        "results": [
            {
                "id": str(r.get("_id")),
                "title": r.get("titulo"),
                "url": r.get("url"),
                "doi": r.get("doi"),
                "year": r.get("publication_date").year if isinstance(r.get("publication_date"), datetime) else r.get("publication_date"),
                "autores": r.get("autores"),
                "categorias": r.get("categorias"),
                "tipo_articulo": r.get("tipo_articulo"),
                "score": r.get("score"),
                "snippet": (r.get("abstract") or "")[:400]
            } for r in results
        ],
        "context_preview": context[:1000],
        "answer": answer
    }

@app.get("/5-docs")
def get_five_docs():
    def _sanitize_doc(doc):
        if not isinstance(doc, dict):
            return doc
        out = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                out[k] = str(v)
            elif isinstance(v, list):
                new_list = []
                for item in v:
                    if isinstance(item, dict):
                        new_list.append(_sanitize_doc(item))
                    elif isinstance(item, ObjectId):
                        new_list.append(str(item))
                    else:
                        new_list.append(item)
                out[k] = new_list
            elif isinstance(v, dict):
                out[k] = _sanitize_doc(v)
            else:
                out[k] = v
        return out

    docs = []
    cursor = col.find({}).limit(5)
    for d in cursor:
        docs.append(_sanitize_doc(d))
    return {"docs": docs}
