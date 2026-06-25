import os
import re
from pathlib import Path
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Resolve paths relative to the project root (two levels up from this file)
ROOT = Path(__file__).parent.parent

DATA_DIRS = {
    str(ROOT / "data" / "indian_supreme_court_pdfs"): "case_law",
    str(ROOT / "data" / "penal_code"): "penal_code",
    str(ROOT / "data" / "code_of_criminal_procedure"): "crpc",
}
CHROMA_PATH = str(ROOT / "storage" / "chroma_db")


# Matches "Section 302", "Sec. 120B", "s. 34", "sections 302" etc.
# Captures the number+optional-letter: "302", "120B", "34"
_SECTION_RE = re.compile(r'\bsec(?:tion)?s?\.?\s*(\d{1,3}[A-Za-z]?)', re.IGNORECASE)


def extract_section_numbers(text: str) -> str:
    """
    Finds all IPC / CrPC section references in a chunk of text and returns
    them as a pipe-separated string stored in ChromaDB metadata.

    Example: "Under Section 302 read with Section 34 IPC..." → "302|34"

    Pipe-separated (not comma) so the retriever can do a simple
    `"302" in metadata["sections"].split("|")` check without false positives
    on sub-strings (e.g. "3" inside "302").
    """
    matches = _SECTION_RE.findall(text)
    seen, unique = set(), []
    for m in matches:
        key = m.strip().upper()
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return "|".join(unique)


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF block by block to preserve order, returning a list of dicts.
    Each dict contains page number and text.
    """
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract blocks of text. Using "blocks" manages multi-column/complex layouts better.
        blocks = page.get_text("blocks")
        # Blocks are tuples: (x0, y0, x1, y1, "lines in block", block_no, block_type)
        if not blocks:
            continue

        # Sort vertically then horizontally
        blocks.sort(key=lambda b: (b[1], b[0]))

        text_content = "\n".join([b[4] for b in blocks if len(b) >= 7 and b[6] == 0])  # 0 means text block

        if text_content.strip():
            pages_data.append({"page": page_num + 1, "text": text_content.strip()})

    return pages_data


def chunk_pdf_data(pages_data, filename):
    """
    Splits each page's text into ~1200-token chunks with ~200-token overlap.

    Larger chunks keep a full statutory clause (section + punishment) or a
    self-contained passage of legal reasoning together; the overlap prevents
    losing context that straddles a chunk boundary.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = []
    for data in pages_data:
        page_chunks = text_splitter.split_text(data["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "page": data["page"],
                    "case_title": filename.replace(".pdf", "").replace("_", " ").title(),
                    # Pipe-separated section numbers found in this chunk, e.g. "302|34".
                    # Empty string if none. Used by the retriever for exact-section lookup.
                    "sections": extract_section_numbers(chunk),
                }
            })
    return chunks


def ingest_pdfs():
    """
    Iterates over PDF directories, extracts, chunks, and stores into ChromaDB.
    """
    all_chunks = []

    for current_dir, source_type in DATA_DIRS.items():
        if not os.path.exists(current_dir):
            print(f"Directory {current_dir} does not exist. Skipping.")
            continue

        pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"No PDFs found in {current_dir}.")
            continue

        for pdf_file in pdf_files:
            print(f"Processing {pdf_file} [{source_type}]...")
            pdf_path = os.path.join(current_dir, pdf_file)
            pages_data = extract_text_from_pdf(pdf_path)
            chunks = chunk_pdf_data(pages_data, pdf_file)

            for c in chunks:
                c["metadata"]["source_type"] = source_type

            all_chunks.extend(chunks)

    print(f"Total chunks generated: {len(all_chunks)}")
    if len(all_chunks) == 0:
        print("No text could be extracted.")
        return

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Storing chunks in local ChromaDB...")
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    # Chroma.from_texts persists the collection to disk; the returned handle
    # isn't needed here since retrieval.py re-opens the store independently.
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},  # cosine distance for sentence embeddings
    )
    print("Ingestion complete! Run the RAG retriever next.")


if __name__ == "__main__":
    ingest_pdfs()
