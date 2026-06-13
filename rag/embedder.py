import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = "index/faiss_store"
MODEL_NAME = "all-mpnet-base-v2"   

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # cosine similarity
    )

def build_and_save(chunks, metadatas):
    print(f"Building FAISS index over {len(chunks)} chunks with all-mpnet-base-v2...")
    embedding_model = get_embedding_model()
    db = FAISS.from_texts(chunks, embedding_model, metadatas=metadatas)
    os.makedirs(INDEX_PATH, exist_ok=True)
    db.save_local(INDEX_PATH)
    print("Index saved.")
    return db

def load_db():
    embedding_model = get_embedding_model()
    db = FAISS.load_local(
        INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("Index loaded.")
    return db

def index_exists():
    return os.path.exists(os.path.join(INDEX_PATH, "index.faiss"))