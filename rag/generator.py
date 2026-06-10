import os
import cohere
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

PROMPTS = {
    "QA": PromptTemplate(
        input_variables=["context", "query"],
        template="""You are a financial assistant.
Answer the question using ONLY the given context.
If the answer is not in context, say: "Not found in context."
Keep your answer precise and factual.

Context:
{context}

Question:
{query}

### Answer:"""
    ),

    "Summary": PromptTemplate(
        input_variables=["context", "query"],
        template="""You are a financial analyst.
Summarize the key information from the context in 3-4 clear lines.
Use ONLY the provided context. Do NOT add new information.

Context:
{context}

### Summary:"""
    ),

    "Extraction": PromptTemplate(
        input_variables=["context", "query"],
        template="""You are a financial assistant.
Extract exactly 4 key points related to: {query}

STRICT RULES:
- Use ONLY the given context
- Output EXACTLY 4 bullet points
- Keep each point short and factual

Context:
{context}

### Answer:"""
    ),

    "Rewrite": PromptTemplate(
        input_variables=["context", "query"],
        template="""Rewrite the following financial content in simple, beginner-friendly English.
3-4 lines. Plain language. Focus on meaning.

Content:
{context}

### Rewritten:"""
    )
}


def get_api_key():
    try:
        return st.secrets["COHERE_API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.environ.get("COHERE_API_KEY") 
    
def generate(query, docs, mode="QA"):
    api_key = get_api_key()
    if not api_key:
        return "❌ COHERE_API_KEY not set."

    client = cohere.ClientV2(api_key=api_key)
    context = "\n\n".join(docs)
    prompt_template = PROMPTS[mode]
    prompt = prompt_template.format(context=context, query=query)

    try:
        response = client.chat(
            model="command-r-plus-08-2024",  
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.message.content[0].text
        return post_process(raw, mode)
    except Exception as e:
        return f"❌ Cohere error: {e}"


def post_process(output, mode):
    markers = {
        "QA": "### Answer:",
        "Summary": "### Summary:",
        "Extraction": "### Answer:",
        "Rewrite": "### Rewritten:"
    }
    marker = markers[mode]
    answer = output.split(marker)[-1].strip() if marker in output else output.strip()
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    noise = {"you are", "context:", "question:", "rules:", "strict"}
    lines = [l for l in lines if not any(l.lower().startswith(n) for n in noise)]
    return "\n".join(lines[:6])