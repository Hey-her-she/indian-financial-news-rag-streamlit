import pandas as pd
import re
import random
import nltk
nltk.download("punkt", quiet=True)

def load_and_clean(csv_path, sample_size=5000):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["Title", "Description"])
    df["text"] = df["Title"] + " " + df["Description"]

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.strip()

    df["clean_text"] = df["text"].apply(clean_text)
    df = df.drop_duplicates(subset=["clean_text"])
    df["doc_length"] = df["clean_text"].apply(lambda x: len(x.split()))
    df = df[df["doc_length"] >= 20].reset_index(drop=True)

    # Random sample — avoid date bias of taking first N rows
    texts = df["clean_text"].tolist()
    dates = df["Date"].tolist() if "Date" in df.columns else ["unknown"] * len(df)
    sampled_idx = random.sample(range(len(texts)), min(sample_size, len(texts)))

    return (
        [texts[i] for i in sampled_idx],
        [dates[i] for i in sampled_idx]
    )


def chunk_texts(sentences, dates, size=100, overlap=20):
    all_chunks = []
    all_metadata = []

    for text, date in zip(sentences, dates):
        words = text.split()
        for i in range(0, len(words), size - overlap):
            chunk = " ".join(words[i:i + size])
            if len(chunk.split()) >= 15:  # skip tiny tail chunks
                all_chunks.append(chunk)
                all_metadata.append({"date": str(date), "source": text[:80]})

    return all_chunks, all_metadata