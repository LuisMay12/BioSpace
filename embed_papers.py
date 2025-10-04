# embed_papers.py
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://<USER>:<PASS>@<CLUSTER>/<DB>?retryWrites=true&w=majority")
DB_NAME = os.getenv("DB_NAME", "BioSpacedb")
COLL = os.getenv("COLL", "papers")

MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

client = MongoClient(MONGO_URI)
col = client[DB_NAME][COLL]
model = SentenceTransformer(MODEL_NAME)

# qué texto embebemos (ajusta a tu gusto)
def build_text(doc):
    parts = []
    if doc.get("titulo"): parts.append(str(doc["titulo"]))
    if doc.get("abstract"): parts.append(str(doc["abstract"]))
    # si quieres: categorias/tipo_articulo para señal adicional
    if doc.get("categorias"): parts.append("; ".join(doc["categorias"]) if isinstance(doc["categorias"], list) else str(doc["categorias"]))
    if doc.get("tipo_articulo"): parts.append(str(doc["tipo_articulo"]))
    text = ". ".join(p for p in parts if p)
    return text if text else None

# procesa por lotes
BATCH = 200
cursor = col.find({}, {"_id": 1, "titulo": 1, "abstract": 1, "categorias": 1, "tipo_articulo": 1})
batch_docs = []
for d in cursor:
    text = build_text(d)
    if not text:
        continue
    batch_docs.append((d["_id"], text))
    if len(batch_docs) >= BATCH:
        ids, texts = zip(*batch_docs)
        vecs = model.encode(list(texts), batch_size=64, show_progress_bar=False)
        for _id, vec in zip(ids, vecs):
            col.update_one({"_id": _id}, {"$set": {"embedding": vec.tolist(), "embedding_model": MODEL_NAME}})
        batch_docs = []

# resto
if batch_docs:
    ids, texts = zip(*batch_docs)
    vecs = model.encode(list(texts), batch_size=64, show_progress_bar=False)
    for _id, vec in zip(ids, vecs):
        col.update_one({"_id": _id}, {"$set": {"embedding": vec.tolist(), "embedding_model": MODEL_NAME}})

print("Embeddings actualizados.")
