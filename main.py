"""Simple RAG: Gradio + Chroma + Ollama (embeddings and LLM both local)."""
from pathlib import Path

import chromadb
import gradio as gr
import ollama
from pypdf import PdfReader

LLM_MODEL = "llama3.2"            # ollama pull llama3.2
EMBED_MODEL = "nomic-embed-text"  # ollama pull nomic-embed-text
CHUNK_SIZE, OVERLAP, TOP_K = 800, 150, 4

client = chromadb.Client()  # in-memory; use chromadb.PersistentClient("db") to keep data
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

SYSTEM_PROMPT = (
    "Answer the question using ONLY the context below. "
    "If the context does not contain the answer, say "
    "\"I don't know based on the provided documents.\"\n\nContext:\n{context}"
)


# ---------- ingest ----------
def load_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return Path(path).read_text(errors="ignore")


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts).embeddings


def index_files(paths: list[str] | None) -> str:
    if not paths:
        return "No files selected."
    total = 0
    for path in paths:
        name = Path(path).name
        chunks = chunk(load_text(path))
        for i in range(0, len(chunks), 32):  # embed in batches
            batch = chunks[i : i + 32]
            collection.upsert(
                ids=[f"{name}-{i + j}" for j in range(len(batch))],
                documents=batch,
                embeddings=embed(batch),
                metadatas=[{"source": name, "chunk": i + j} for j in range(len(batch))],
            )
        total += len(chunks)
    return f"Indexed {total} chunks from {len(paths)} file(s)."


# ---------- retrieve + generate ----------
def retrieve(question: str, k: int = TOP_K):
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer(message: str, history: list):
    if collection.count() == 0:
        yield "Upload and index some documents first (left sidebar)."
        return

    hits = retrieve(message)
    context = "\n\n---\n\n".join(doc for doc, _ in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": message},
    ]

    text = ""
    for part in ollama.chat(model=LLM_MODEL, messages=messages, stream=True):
        text += part["message"]["content"]
        yield text

    sources = "\n".join(
        f"- **{m['source']}** (chunk {m['chunk']}): {d[:120].replace(chr(10), ' ')}..."
        for d, m in hits
    )
    yield f"{text}\n\n---\n**Sources**\n{sources}"


# ---------- UI ----------
with gr.Blocks(title="Simple RAG") as demo:
    with gr.Sidebar():
        gr.Markdown("### Documents")
        files = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md"], type="filepath")
        status = gr.Textbox(label="Status", interactive=False)
        gr.Button("Index documents", variant="primary").click(index_files, files, status)
    gr.ChatInterface(answer, title="Chat with your documents")

if __name__ == "__main__":
    demo.launch()