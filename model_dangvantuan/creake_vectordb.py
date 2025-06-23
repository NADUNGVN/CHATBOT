import os, pdfplumber
import pandas as pd
from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModel
from chromadb import PersistentClient

# ============ CẤU HÌNH ============
PDF_DIR = r"E:\WORK\project\chatbot_RAG\data\1_pdf_6_files"
PERSIST_DIR = r"E:\WORK\project\chatbot_RAG\model_dangvantuan\chroma_db"
CHUNK_SIZE = 2048
OVERLAP = 200
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"

# ============ KHỞI TẠO ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()  # tắt dropout, tăng ổn định khi sinh embedding

chroma_client = PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name="rag_chunks")

# ============ EMBEDDING FUNCTION ============
def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean Pooling
    return embeddings[0].cpu().numpy()

# ============ PDF TO CHUNKS ============
def load_pdf_tokens(pdf_path):
    full_tokens, page_ranges, token_offset = [], [], 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            txt = page.extract_text() or ""
            ids = tokenizer.encode(txt, add_special_tokens=False)
            full_tokens.extend(ids)
            page_ranges.append({
                "page": page_num,
                "start": token_offset,
                "end": token_offset + len(ids)
            })
            token_offset += len(ids)
    return full_tokens, page_ranges

def chunk_tokens(tokens, page_ranges, chunk_size=2048, overlap=200, start_id=1):
    chunks, cur = [], start_id
    pos = 0
    while pos < len(tokens):
        end = min(pos + chunk_size, len(tokens))
        ids = tokens[pos:end]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        pages = [p["page"] for p in page_ranges if p["end"] > pos and p["start"] < end]
        chunks.append({
            "chunk_id": cur,
            "start_tok": pos,
            "end_tok": end,
            "pages": pages,
            "token_cnt": len(ids),
            "content": text
        })
        cur += 1
        pos = end - overlap if end - overlap > pos else end
    return chunks, cur

# ============ QUY TRÌNH CHÍNH ============
def process_all_pdfs(pdf_dir):
    all_chunks, next_id = [], 1
    for fn in os.listdir(pdf_dir):
        if not fn.lower().endswith(".pdf"):
            continue
        print(f"📄 Đang xử lý: {fn}")
        path = os.path.join(pdf_dir, fn)
        tokens, ranges = load_pdf_tokens(path)
        chunks, next_id = chunk_tokens(tokens, ranges, CHUNK_SIZE, OVERLAP, start_id=next_id)
        for c in chunks:
            c["file"] = fn
        all_chunks.extend(chunks)
    return all_chunks

def add_to_vectordb(chunks):
    print("⚙️  Đang sinh embedding và lưu vào VectorDB...")
    for chunk in tqdm(chunks):
        try:
            embedding = get_embedding(chunk["content"])
            collection.add(
                documents=[chunk["content"]],
                embeddings=[embedding],
                metadatas=[{
                    "file": chunk["file"],
                    "chunk_id": chunk["chunk_id"],
                    "pages": ", ".join(map(str, chunk["pages"])),
                    "start_tok": chunk["start_tok"],
                    "end_tok": chunk["end_tok"],
                    "token_cnt": chunk["token_cnt"]
                }],
                ids=[str(chunk["chunk_id"])]
            )
        except Exception as e:
            print(f"Lỗi ở chunk {chunk['chunk_id']} - {chunk['file']}: {e}")
    print("✅ Hoàn tất!")

# ============ CHẠY ============
if __name__ == "__main__":
    chunks = process_all_pdfs(PDF_DIR)
    add_to_vectordb(chunks)
