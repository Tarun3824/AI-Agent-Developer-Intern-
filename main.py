from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
doc_store = []

class Document(BaseModel):
    id: str
    content: str

@app.post("/index")
def index_document(doc: Document):
    embedding = model.encode([doc.content])
    index.add(np.array(embedding).astype("float32"))
    doc_store.append({"id": doc.id, "content": doc.content})
    return {"status": "indexed", "id": doc.id}
