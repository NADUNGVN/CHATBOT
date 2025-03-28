import os
import logging
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import time
from difflib import SequenceMatcher
from config.settings import Config

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'chatbot_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. Tải các vector database đã tạo
def load_retrievers(base_persist_directory):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="dangvantuan/vietnamese-document-embedding",
            model_kwargs={"trust_remote_code": True}
        )
        
        categories = ["trong_nuoc", "nuoc_ngoai", "dac_biet", "lien_thong"]
        retrievers = {}
        vectordbs = {}
        
        logger.info(f"Bắt đầu tải retrievers từ {base_persist_directory}")
        
        for category in categories:
            try:
                persist_directory = os.path.join(base_persist_directory, category)
                if os.path.exists(persist_directory):
                    db = Chroma(
                        persist_directory=persist_directory, 
                        embedding_function=embeddings
                    )
                    vectordbs[category] = db
                    retrievers[category] = db.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": 4, "score_threshold": 0.5}
                    )
                    logger.info(f"Đã tải thành công retriever cho collection '{category}'")
                else:
                    logger.warning(f"Không tìm thấy thư mục cho collection '{category}'")
            except Exception as e:
                logger.error(f"Lỗi khi tải retriever cho collection '{category}': {str(e)}")
        
        return vectordbs, retrievers
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi tải retrievers: {str(e)}")
        raise

# 2. Xác định collection phù hợp với câu hỏi
def determine_collection(query):
    query_lower = query.lower()
    
    # Thêm từ khóa và trọng số cho việc phân loại
    keywords = {
        "nuoc_ngoai": ["nước ngoài", "quốc tế", "ngoại quốc", "người nước ngoài", "ngoại kiều"],
        "lien_thong": ["liên thông", "kết hợp", "đồng thời", "kết nối"],
        "dac_biet": ["lưu động", "đăng ký lại", "đặc biệt", "thay đổi", "cải chính"]
    }
    
    scores = {category: 0 for category in keywords}
    for category, words in keywords.items():
        for word in words:
            if word in query_lower:
                scores[category] += 1
    
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    return "trong_nuoc"

# Hàm mới: Tìm và đánh dấu thông tin trích xuất từ tài liệu nguồn
def highlight_source_information(answer, source_documents):
    highlighted_answer = answer
    source_info = []
    
    for i, doc in enumerate(source_documents):
        content = doc.page_content
        source_file = doc.metadata.get('source', 'Unknown')
        page_number = doc.metadata.get('page', 'Unknown')
        
        s = SequenceMatcher(None, content, answer)
        matches = s.get_matching_blocks()
        
        significant_matches = [match for match in matches if match.size >= 10]
        
        for match in significant_matches:
            source_text = content[match.a:match.a + match.size]
            if source_text in answer:
                source_info.append({
                    "text": source_text,
                    "source": os.path.basename(source_file),
                    "page": page_number,
                    "doc_index": i
                })
    
    source_info.sort(key=lambda x: len(x["text"]), reverse=True)
    
    return highlighted_answer, source_info

# 3. Tạo chatbot với khả năng chọn collection phù hợp
def build_rag_chatbots(vectordbs):
    try:
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.5,
            openai_api_key="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"
        )
        
        # Tạo memory riêng cho mỗi category
        memories = {}
        qa_chains = {}
        
        for category in vectordbs.keys():
            memories[category] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            retriever = vectordbs[category].as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.45}
            )
            
            qa_chains[category] = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memories[category],
                return_source_documents=True,
                verbose=True
            )
        
        return qa_chains

    except Exception as e:
        logger.error(f"Error building RAG chatbots: {str(e)}")
        raise

# 4. Xử lý câu hỏi và trả lời
def process_query(query, qa_chains, retrievers):
    try:
        start_time = time.time()
        logger.info(f"Xử lý câu hỏi: {query}")
        
        collection = determine_collection(query)
        logger.info(f"Đã xác định collection: {collection}")
        
        if collection not in qa_chains:
            logger.warning(f"Không tìm thấy collection '{collection}', thử tìm collection thay thế")
            results = []
            for category, retriever in retrievers.items():
                try:
                    docs = retriever.get_relevant_documents(query)
                    if docs:
                        results.extend([(category, len(docs))])
                        logger.debug(f"Tìm thấy {len(docs)} tài liệu trong collection '{category}'")
                except Exception as e:
                    logger.error(f"Lỗi khi tìm kiếm trong collection '{category}': {str(e)}")
            
            if results:
                collection = max(results, key=lambda x: x[1])[0]
                logger.info(f"Đã chọn collection thay thế: {collection}")
            else:
                collection = "trong_nuoc"
                logger.warning("Không tìm thấy kết quả phù hợp, sử dụng collection mặc định")
        
        result = qa_chains[collection].invoke({"question": query})
        processing_time = time.time() - start_time
        logger.info(f"Đã xử lý câu hỏi trong {processing_time:.2f} giây")
        
        return {
            "answer": result.get("answer", ""),
            "collection": collection,
            "source_documents": result.get("source_documents", []),
            "source_info": highlight_source_information(result.get("answer", ""), 
                                                      result.get("source_documents", []))[1],
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Lỗi khi xử lý câu hỏi: {str(e)}", exc_info=True)
        return {
            "answer": f"Xin lỗi, có lỗi xảy ra: {str(e)}",
            "collection": None,
            "source_documents": [],
            "source_info": [],
            "processing_time": time.time() - start_time
        }

# 5. Tích hợp vào giao diện người dùng
def run_chatbot(qa_chains, retrievers):
    print("Chào mừng bạn đến với RAG Chatbot! Gõ 'exit' để thoát.")
    while True:
        query = input("\nBạn: ")
        if query.lower() == "exit":
            break

        result = process_query(query, qa_chains, retrievers)
        
        print(f"\nBot [{result['collection']}]: {result['answer']}")
        
        if 'processing_time' in result:
            print(f"\nThời gian xử lý: {result['processing_time']:.2f} giây")
        
        source_info = result.get("source_info", [])
        if source_info:
            print("\nThông tin được trích xuất từ các nguồn sau:")
            for i, info in enumerate(source_info):
                print(f"{i + 1}. Đoạn: \"{info['text']}\"")
                print(f"   - Nguồn: {info['source']}, Trang: {info['page']}")
        
        sources = result.get("source_documents", [])
        if sources:
            print("\nNguồn tham khảo:")
            for i, doc in enumerate(sources):
                source_file = doc.metadata.get('source', 'Unknown')
                page_number = doc.metadata.get('page', 'Unknown')
                print(f"{i + 1}. File: {os.path.basename(source_file)}, Trang: {page_number}")

def main():
    base_persist_directory = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\src_main_2\chroma_pdf_db_2\Ho_Tich"
    vectordbs, retrievers = load_retrievers(base_persist_directory)
    
    if not vectordbs:
        print("Không tìm thấy collection nào. Vui lòng chạy create_vectordb.py trước.")
        return
    
    qa_chains = build_rag_chatbots(vectordbs)
    run_chatbot(qa_chains, retrievers)

if __name__ == "__main__":
    main()