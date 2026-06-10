from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
RERANKER = None
def get_reranker():
    global RERANKER
    if RERANKER is None:
        print("Loading CrossEncoder reranker...")
        RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return RERANKER

def retrieve_and_rerank(query, db, top_k=5, fetch_k=20):
    # Stage 1: LangChain retriever
    retriever = db.as_retriever(search_kwargs={"k": fetch_k})
    candidate_docs = retriever.invoke(query)
    candidate_texts = [doc.page_content for doc in candidate_docs]

    # Stage 2: CrossEncoder reranking
    reranker = get_reranker()
    pairs = [(query, text) for text in candidate_texts]
    raw_scores = reranker.predict(pairs)

    # Normalize raw logits to 0-1 using sigmoid
    normalized_scores = torch.sigmoid(torch.tensor(raw_scores)).tolist()

    reranked = sorted(
        zip(candidate_texts, normalized_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [text for text, _ in reranked[:top_k]], [score for _, score in reranked[:top_k]]