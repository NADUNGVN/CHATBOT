import os
import shutil
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load and process data from multiple PDF files
def load_pdf_documents(pdf_directory):
    loader = DirectoryLoader(
        pdf_directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    return documents

# 2. Split documents into manageable chunks for PDF
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Classify documents based on filename or content
def classify_document(document):
    source = document.metadata.get("source", "").lower()
    content = document.page_content.lower()
    
    if "yếu tố nước ngoài" in source or "nước ngoài" in source:
        return "nuoc_ngoai"
    elif "liên thông" in source:
        return "lien_thong"
    elif any(term in source for term in ["lưu động", "đăng ký lại", "người đã có hồ sơ", "thay đổi", "cải chính", "xác định lại", "bổ sung", "kết hợp"]):
        return "dac_biet"
    else:
        return "trong_nuoc"

# 3. Create embeddings and store vectors for each collection
def create_vectordbs(chunks, base_persist_directory):
    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True}
    )
    
    categorized_chunks = {
        "trong_nuoc": [],
        "nuoc_ngoai": [],
        "dac_biet": [],
        "lien_thong": []
    }
    
    for chunk in chunks:
        category = classify_document(chunk)
        categorized_chunks[category].append(chunk)
    
    parent_collection = "Ho_Tich"
    vectordbs = {}
    parent_directory = os.path.join(base_persist_directory, parent_collection)
    os.makedirs(parent_directory, exist_ok=True)

    for category, category_chunks in categorized_chunks.items():
        if category_chunks:
            persist_directory = os.path.join(parent_directory, category)
            vectordbs[category] = Chroma.from_documents(
                documents=category_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            print(f"Created vector database '{category}' in parent collection '{parent_collection}' with {len(category_chunks)} chunks")

            files = set(chunk.metadata.get("source", "Unknown") for chunk in category_chunks)
            print(f"Files in collection '{category}':")
            for file in files:
                print(f"- {file}")
    
    return vectordbs

# 4. Load existing vector databases (to avoid reprocessing PDFs)
def load_vectordbs(base_persist_directory, embeddings):
    vectordbs = {}
    categories = ["trong_nuoc", "nuoc_ngoai", "dac_biet", "lien_thong"]
    
    for category in categories:
        persist_directory = os.path.join(base_persist_directory, category)
        if os.path.exists(persist_directory):
            vectordbs[category] = Chroma(
                persist_directory=persist_directory, 
                embedding_function=embeddings
            )
    
    return vectordbs

# Main function
def main():
    pdf_directory = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\src_main_2\data\pdf"
    base_persist_directory = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\src_main_2\chroma_pdf_db_2"
    
    if os.path.exists(base_persist_directory):
        shutil.rmtree(base_persist_directory)
    
    os.makedirs(base_persist_directory, exist_ok=True)
    
    documents = load_pdf_documents(pdf_directory)
    chunks = split_documents(documents)
    vectordbs = create_vectordbs(chunks, base_persist_directory)

if __name__ == "__main__":
    main()