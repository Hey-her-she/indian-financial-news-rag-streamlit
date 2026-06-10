# 📈 Indian Financial News RAG System

A full-stack Retrieval-Augmented Generation (RAG) system for semantic search and question answering over 50,000 Indian financial news articles (2003–2020).

---

## 🧠 Pipeline Overview

```
Query
  │
  ▼
LangChain + FAISS Retriever (top 20 candidates)
  │
  ▼
CrossEncoder Reranker (top 5 most relevant)
  │
  ▼
Cohere command-r-plus (answer generation)
  │
  ▼
Post-processed Answer
```

| Stage | Model / Tool | Role |
|---|---|---|
| Embedding | `all-mpnet-base-v2` | Converts text into semantic vectors |
| Retrieval | LangChain + FAISS `IndexFlatIP` | Finds top 20 similar chunks |
| Reranking | CrossEncoder `ms-marco-MiniLM-L-6-v2` | Re-scores candidates, keeps top 5 |
| Generation | Cohere `command-r-plus-08-2024` | Produces answer from context + prompt |

---

## 🗂️ Dataset

**Indian Financial News Articles (2003–2020)**
- ~50,000 articles
- Columns: `Date`, `Title`, `Description`
- Domain-specific vocabulary: RBI, NBFC, repo rate, gilt, PSU
- Source: Kaggle

### Preprocessing
- Dropped missing values and duplicates (793 found)
- Combined Title + Description into a single text field
- Lowercased, removed symbols, normalized spaces
- Filtered documents shorter than 20 words (removed 4,671)
- Final corpus: **43,826 documents**
- Random sample of 5,000 used for indexing to avoid date bias

---

## 🏗️ Project Structure

```
financial_rag/
├── app.py                  # Streamlit frontend
├── build_index.py          # Run once to build and save FAISS index
├── rag/
│   ├── __init__.py
│   ├── preprocessor.py     # Cleaning + overlapping chunking
│   ├── embedder.py         # LangChain FAISS build + save/load
│   ├── retriever.py        # LangChain retriever + CrossEncoder rerank
│   └── generator.py        # LangChain PromptTemplate + Cohere generation
├── data/
│   └── IndianFinancialNews.csv
├── .streamlit/
│   └── secrets.toml        # API keys (not committed)
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Features

- **4 Response Modes** — QA, Summarization, Extraction, Rewriting
- **Two-stage retrieval** — FAISS bi-encoder + CrossEncoder reranker
- **Overlapping chunks** — 100-word chunks with 20-word overlap to preserve context boundaries
- **Normalized relevance scores** — CrossEncoder logits mapped to 0–1 via sigmoid
- **Evaluation tab** — Rate answers 1–5 stars, view session history, export ratings as CSV
- **Persistent index** — FAISS index saved after first build for instant subsequent loads

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/financial-rag.git
cd financial-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create `.streamlit/secrets.toml`:
```toml
COHERE_API_KEY = "your_key_here"
```

Or create a `.env` file:
```
COHERE_API_KEY=your_key_here
```

Get a free Cohere API key at [dashboard.cohere.com](https://dashboard.cohere.com)

### 4. Build the index (first time only)
```bash
python build_index.py
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
sentence-transformers
faiss-cpu
langchain
langchain-community
langchain-core
nltk
cohere
torch
python-dotenv
```

---

## 🔍 Example Queries

| Mode | Query |
|---|---|
| QA | `What steps did RBI take to control inflation?` |
| QA | `Which banks were affected by the NPA crisis?` |
| Extraction | `What are the key issues related to bad loans in Indian banks?` |
| Summary | `Summarize the state of Indian banking sector between 2015 and 2018` |
| Rewrite | `Explain what a repo rate is in simple terms` |

---

## 🧪 Evaluation

The app includes a built-in evaluation tab that allows you to:
- Rate each answer from 1–5 stars
- View average rating across the session
- See rating distribution as a bar chart
- Export all rated queries and answers as a CSV

---

## 🛠️ Tech Stack

- **Frontend** — Streamlit
- **Embeddings** — `sentence-transformers` (all-mpnet-base-v2)
- **Vector Store** — FAISS via LangChain
- **Reranker** — CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **Prompt Management** — LangChain PromptTemplate
- **LLM** — Cohere command-r-plus-08-2024
- **Deployment** — Streamlit Community Cloud

---

## 📄 License

MIT
