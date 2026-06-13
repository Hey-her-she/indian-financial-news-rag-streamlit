from rag.preprocessor import load_and_clean, chunk_texts
from rag.embedder import build_and_save, index_exists
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if index_exists():
    print("Index already exists. Delete index/faiss_store to rebuild.")
else:
    print("Loading and cleaning data...")
    texts, dates = load_and_clean("data/IndianFinancialNews.csv", sample_size=50000)
    print(f"Sampled {len(texts)} documents.")

    print("Chunking...")
    chunks, metadatas = chunk_texts(texts, dates)
    print(f"Created {len(chunks)} chunks.")

    build_and_save(chunks, metadatas)
    print("Done! Run: streamlit run app.py")