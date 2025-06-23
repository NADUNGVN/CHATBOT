import os
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.chroma import Chroma
from chromadb import PersistentClient
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA  
from langchain.prompts import PromptTemplate  
import time
import openpyxl
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

class DangVanTuanEmbedding(Embeddings):
    def __init__(self, model_name="dangvantuan/vietnamese-document-embedding", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)

    def embed_query(self, text: str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].cpu().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

VIETNAMESE_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên giải đáp thủ tục hành chính bằng tiếng Việt. Hãy trả lời ngắn gọn, rõ ràng, chính xác và lịch sự. Nếu không có thông tin phù hợp, hãy nói rõ rằng bạn không tìm thấy dữ liệu liên quan."""

# 1. Tải vector database đã tạo
def load_vectordb(persist_directory):
    embeddings = DangVanTuanEmbedding(
        model_name="dangvantuan/vietnamese-document-embedding",
        device="cpu"
    )
    
    if os.path.exists(persist_directory):
        chroma_client = PersistentClient(path=persist_directory)
        db = Chroma(
            client=chroma_client,
            collection_name="rag_chunks",
            embedding_function=embeddings
        )
        print(f"✅ Đã tải vector database từ: {persist_directory}")
        return db
    
    print("⚠️ Không tìm thấy thư mục chứa VectorDB.")
    return None

# 2. Tạo chatbot với RAG
def build_rag_chatbot(vectordb):
    try:
        # Khởi tạo language model với Groq
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.5,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=2000
        )
        
        # Create custom prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=f"""{VIETNAMESE_SYSTEM_PROMPT}

        Dựa vào các thông tin sau, hãy trả lời câu hỏi:

        Thông tin: {{context}}

        Câu hỏi: {{question}}

        Trả lời bằng tiếng Việt:"""
        )
        
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3
            }
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template,
                "verbose": True
            }
        )
        return qa_chain
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo chatbot: {str(e)}")
        return None

# 3. Xử lý câu hỏi và trả lời
def process_query(query, qa_chain):
    start_time = time.time()
    
    try:
        result = qa_chain.invoke({
            "query": query
        })
        elapsed_time = time.time() - start_time

        source_docs = result.get("source_documents", [])
        
        answer = result.get("result", "").strip()
        if not answer or answer == "Tôi không tìm thấy thông tin về vấn đề này.":
            answer = "Xin lỗi, tôi không tìm thấy thông tin đủ liên quan để trả lời câu hỏi của bạn."
        
        return {
            "answer": answer,
            "source_documents": source_docs,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        print(f"Lỗi khi xử lý câu hỏi: {str(e)}")
        elapsed_time = time.time() - start_time
        error_message = "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."

# 4. Tích hợp vào giao diện người dùng
def run_chatbot(qa_chain):
    print("Chào mừng bạn đến với RAG Chatbot! Gõ 'exit' để thoát.")
    while True:
        query = input("\nBạn: ")
        if query.lower() == "exit":
            break
        result = process_query(query, qa_chain)
        # Hiển thị kết quả
        print(f"\nBot: {result['answer']}")
        print(f"\n⏱️ Thời gian xử lý: {result['elapsed_time']:.2f} giây")
        # Hiển thị nguồn tài liệu
        sources = result.get("source_documents", [])
        if sources:
            print("\nNguồn tham khảo:")
            for i, doc in enumerate(sources):
                source_file = os.path.basename(doc.metadata.get('file', 'Unknown'))
                chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                print(f"{i + 1}. File: {source_file}, Chunk ID: {chunk_id}")

def main():
    # Thư mục chứa vector database
    persist_directory = r"E:\WORK\project\chatbot_RAG\model_dangvantuan\chroma_db"
    
    vectordb = load_vectordb(persist_directory)
    
    # Kiểm tra xem có tải được vector database không
    if not vectordb:
        print("Không tìm thấy vector database. Vui lòng chạy create_vectordb.py trước.")
        return
    
    # Tạo chatbot
    qa_chain = build_rag_chatbot(vectordb)
    
    # Kiểm tra xem qa_chain có được khởi tạo thành công không
    if qa_chain is None:
        print("Không thể khởi tạo chatbot. Vui lòng kiểm tra lại cấu hình và API key.")
        return
        
    run_chatbot(qa_chain)

if __name__ == "__main__":
    main()