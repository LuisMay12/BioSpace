# app.py
import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from sentence_transformers import SentenceTransformer
from urllib.parse import quote_plus

# (Opcional) LLM para generación
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://luismayrdz_db_user:tNXwmDPBL4hwyhpu@biospacedb.sgh5mic.mongodb.net/?retryWrites=true&w=majority&appName=BioSpacedb")
# DB_NAME = os.getenv("DB_NAME", "BioSpacedb")
# COLL = os.getenv("COLL", "papers")
# ATLAS_VECTOR_INDEX = os.getenv("ATLAS_VECTOR_INDEX", "default")  # nombre del índice que creaste
# EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# client = MongoClient(MONGO_URI)
# col = client[DB_NAME][COLL]

# ===== Configuración general =====
DB_NAME = os.getenv("DB_NAME", "BioSearch")
COLL = os.getenv("COLL", "papers")
ATLAS_VECTOR_INDEX = os.getenv("ATLAS_VECTOR_INDEX", "default")  # cambia si tu índice tiene otro nombre
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ===== Credenciales Mongo Atlas =====
USER = os.getenv("MONGO_USER", "luismayrdz_db_user")
PASSWORD = os.getenv("MONGO_PASS", "tNXwmDPBL4hwyhpu")  # ⚠️ idealmente guárdala en .env
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

class RAGQuery(BaseModel):
    query: str
    k: int = 6
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    autores: Optional[List[str]] = None      # exact match (case sensitive/insensitive según tus datos)
    categorias: Optional[List[str]] = None   # exact match
    tipo_articulo: Optional[str] = None
    generate: bool = False                   # si quieres que llame LLM
    max_context_chars: int = 3000            # para controlar el tamaño del prompt

def _build_filter(q: RAGQuery):
    # Construye un filtro Atlas Search $vectorSearch + filtros metadata con $match posterior
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
    # Devuelve un pequeño texto con cita
    title = doc.get("titulo") or ""
    pid = doc.get("_id")
    year = doc.get("year")
    doi = doc.get("doi")
    url = doc.get("url")
    return f"[{pid} - {year}] {title} (DOI: {doi}) {url or ''}".strip()

def _compose_context(docs: List[dict], max_chars: int):
    # concatena títulos + trozos del abstract hasta llenar max_chars
    parts = []
    total = 0
    for d in docs:
        head = f"### {_format_citation(d)}\n"
        body = (d.get("abstract") or "")[:800]  # corta por sanidad
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
        # Fallback: devuelve un "extractive summary" simple (sin LLM)
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
    # 1) Embedding de la consulta
    q_vec = emb_model.encode(q.query).tolist()
    print(f"Query vector (len={len(q_vec)}): {q_vec[:5]}...")
    
    print(f"Que mierda es col {col.count_documents({})}")
    
    # checking how many docs has those shitttt
    # 1) ¿Cuántos docs tienen vector?
    print("Docs con embedding:", col.count_documents({"embedding": {"$exists": True, "$type": "array"}}))

    # 2) Mira 1 doc con embedding y su dimensión
    doc = col.find_one({"embedding": {"$exists": True}}, {"titulo":1, "embedding": {"$slice": 5}})
    if doc:
        full = col.find_one({"_id": doc["_id"]}, {"embedding":1})
        print("Ejemplo título:", doc.get("titulo"))
        print("Len embedding:", len(full["embedding"]) if full and full.get("embedding") else None)

    # 3) ¿Qué valores reales tienen 'categorias'?
    print("Algunos 'categorias':", col.distinct("categorias")[:10])  # si es array, te mostrará strings
    print("Distinct 'tipo_articulo':", col.distinct("tipo_articulo")[:10])


    # 2) Vector Search en Atlas (KNN) + filtros con $match
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
            "titulo": 1, "abstract": 1, "url": 1, "year": 1, "doi": 1,
            "autores": 1, "categorias": 1, "tipo_articulo": 1,
            "score": {"$meta": "vectorSearchScore"}
        }},
    ]

    # Aplica filtros después del vectorSearch (o usa "filter" en vectorSearch si tienes Atlas > 7.0)
    match = _build_filter(q)
    if match:
        pipeline.append({"$match": match})

    results = list(col.aggregate(pipeline))

    # 3) Armar contexto y (opcional) preguntar al LLM
    context = _compose_context(results, q.max_context_chars)
    answer = _generate_answer(q.query, context) if q.generate else None

    # 4) Respuesta con snippets y metadatos (citas)
    return {
        "query": q.query,
        "count": len(results),
        "results": [
            {
                "id": str(r.get("_id")),
                "title": r.get("titulo"),
                "url": r.get("url"),
                "doi": r.get("doi"),
                "year": r.get("year"),
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
    # Helper to convert non-JSON-serializable Mongo types (ObjectId) recursively
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
    # Project a small set of fields to keep response compact; adjust as needed
    cursor = col.find({}).limit(5)
    for d in cursor:
        docs.append(_sanitize_doc(d))
    return {"docs": docs}
