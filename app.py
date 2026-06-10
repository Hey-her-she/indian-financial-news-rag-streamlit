import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from rag.embedder import load_db, index_exists
from rag.retriever import retrieve_and_rerank
from rag.generator import generate
import pandas as pd

st.set_page_config(page_title="Financial News RAG", page_icon="📈", layout="wide")
st.title("📈 Indian Financial News RAG")
st.caption("Powered by LangChain · all-mpnet-base-v2 · FAISS · CrossEncoder · Mistral")

# --- Load index ---
@st.cache_resource
def load():
    if not index_exists():
        from rag.preprocessor import load_and_clean, chunk_texts
        from rag.embedder import build_and_save
        import random
        with st.spinner("⚙️ Building index for the first time — takes a few minutes..."):
            texts, dates = load_and_clean("data/IndianFinancialNews.csv")
            pairs = list(zip(texts, dates))
            sampled = random.sample(pairs, min(2000, len(pairs)))
            texts_s, dates_s = zip(*sampled)
            chunks, metadatas = chunk_texts(texts_s, dates_s)
            return build_and_save(chunks, metadatas)
    return load_db()

db = load()

# --- Session state init ---
if "history" not in st.session_state:
    st.session_state.history = []  
    # each entry: {query, answer, mode, docs, scores, rating}

if "pending_rating" not in st.session_state:
    st.session_state.pending_rating = None  
    # holds last result waiting to be rated

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.selectbox("Response mode", ["QA", "Summary", "Extraction", "Rewrite"])
top_k = st.sidebar.slider("Final docs after reranking", 3, 10, 5)
fetch_k = st.sidebar.slider("Candidates for reranker", 10, 30, 20)
show_docs = st.sidebar.checkbox("Show retrieved documents", value=True)
show_scores = st.sidebar.checkbox("Show relevance scores", value=True)

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Search", "📊 Evaluation"])

# ==================== SEARCH TAB ====================
with tab1:
    query = st.text_input(
        "Ask a question:",
        placeholder="e.g. What were the key issues with Indian banking loans?"
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("🔍 Retrieving with LangChain + FAISS..."):
            docs, scores = retrieve_and_rerank(query, db, top_k=top_k, fetch_k=fetch_k)

        with st.spinner("🤖 Generating with Mistral..."):
            answer = generate(query, docs, mode)

        # Store as pending — waiting for rating
        st.session_state.pending_rating = {
            "query": query,
            "answer": answer,
            "mode": mode,
            "docs": docs,
            "scores": scores,
            "rating": None
        }

    # Show pending result + rating widget
    if st.session_state.pending_rating:
        result = st.session_state.pending_rating

        st.markdown("### Answer")
        st.success(result["answer"])

        # Rating widget
        st.markdown("**Rate this answer:**")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if col.button(f"{'⭐' * (i+1)}", key=f"star_{i+1}"):
                result["rating"] = i + 1
                st.session_state.history.append(result.copy())
                st.session_state.pending_rating = None
                st.toast(f"Rated {i+1}/5 — saved to Evaluation tab!", icon="✅")
                st.rerun()

        # Skip rating option
        if st.button("Skip rating"):
            result["rating"] = None
            st.session_state.history.append(result.copy())
            st.session_state.pending_rating = None
            st.rerun()

        # Show retrieved docs
        if show_docs:
            st.markdown("### Retrieved Documents")
            for i, (doc, score) in enumerate(zip(result["docs"], result["scores"])):
                label = f"Document {i+1}"
                if show_scores:
                    label += f"  —  relevance score: `{score:.3f}`"
                with st.expander(label):
                    st.write(doc)

# ==================== EVALUATION TAB ====================
with tab2:
    st.markdown("## 📊 Session Evaluation")

    rated = [h for h in st.session_state.history if h["rating"] is not None]
    all_entries = st.session_state.history

    if not all_entries:
        st.info("No queries yet. Go to the Search tab and ask something!")
    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries", len(all_entries))
        col2.metric("Rated Queries", len(rated))
        avg = sum(h["rating"] for h in rated) / len(rated) if rated else 0
        col3.metric("Average Rating", f"{avg:.1f} / 5" if rated else "—")

        # Rating distribution bar chart
        if rated:
            st.markdown("### Rating Distribution")
            rating_counts = {i: 0 for i in range(1, 6)}
            for h in rated:
                rating_counts[h["rating"]] += 1
            dist_df = pd.DataFrame({
                "Stars": [f"{'⭐' * k}" for k in rating_counts],
                "Count": list(rating_counts.values())
            })
            st.bar_chart(dist_df.set_index("Stars"))

        # Query history table
        st.markdown("### Query History")
        for i, entry in enumerate(reversed(all_entries)):
            rating_display = f"{'⭐' * entry['rating']}" if entry["rating"] else "Not rated"
            with st.expander(f"Q{len(all_entries)-i}: {entry['query'][:80]}  —  {rating_display}"):
                st.markdown(f"**Mode:** `{entry['mode']}`")
                st.markdown(f"**Rating:** {rating_display}")
                st.markdown("**Answer:**")
                st.success(entry["answer"])

        # Export button
        if rated:
            st.markdown("### Export")
            export_df = pd.DataFrame([{
                "Query": h["query"],
                "Mode": h["mode"],
                "Answer": h["answer"],
                "Rating": h["rating"]
            } for h in rated])
            csv = export_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download ratings as CSV",
                data=csv,
                file_name="rag_evaluation.csv",
                mime="text/csv"
            )