import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
from langchain_together import ChatTogether, TogetherEmbeddings
import logging
import re

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_question(question):
    """
    Phân tích câu hỏi để xác định lĩnh vực, tên thủ tục, và yêu cầu.
    Ví dụ câu hỏi: "Các bước thực hiện thủ tục đăng ký khai tử?"
    """
    # Tìm lĩnh vực
    if "hộ tịch" in question.lower():
        field = "Hộ tịch"
    else:
        field = "Không xác định"
    
    # Tìm tên thủ tục
    match = re.search(r"(đăng ký khai tử|đăng ký lại khai tử|các thủ tục|thủ tục đăng ký)", question.lower())
    if match:
        procedure_name = match.group(0).capitalize()
    else:
        procedure_name = "Không xác định"

    # Tìm yêu cầu
    if "các bước thực hiện" in question.lower() or "cách thực hiện" in question.lower():
        request = "Cách thực hiện"
    else:
        request = "Không xác định"

    return field, procedure_name, request

def check_system():
    try:
        # 🔹 Load API keys từ .env
        load_dotenv()
        ZILLIZ_URI = f"https://{os.getenv('ZILLIZ_HOST')}"
        ZILLIZ_TOKEN = os.getenv("ZILLIZ_API_KEY")
        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY_2")

        # 🔹 Kiểm tra kết nối Together AI
        print("\n🤖 Kiểm tra Together AI...")
        try:
            # Khởi tạo mô hình chat
            chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                together_api_key=TOGETHER_API_KEY
            )
            
            # Khởi tạo mô hình embedding
            embedding_model = TogetherEmbeddings(
                model="togethercomputer/m2-bert-80M-32k-retrieval",
                together_api_key=TOGETHER_API_KEY
            )
            print("✅ Kết nối Together AI thành công!")
        except Exception as e:
            print(f"❌ Lỗi kết nối Together AI: {str(e)}")
            return

        # 🔹 Kết nối đến Zilliz Cloud
        print("\n🔄 Đang kết nối đến Zilliz Cloud...")
        try:
            connections.connect(
                alias="default",
                uri=ZILLIZ_URI,
                token=ZILLIZ_TOKEN
            )
            print("✅ Kết nối Zilliz Cloud thành công!")
        except Exception as e:
            print(f"❌ Lỗi kết nối Zilliz: {str(e)}")
            return

        # 🔹 Kiểm tra collections
        collections = utility.list_collections()
        if not collections:
            print("⚠️ Không có collection nào!")
            return
        print(f"📊 Collections hiện có: {collections}")

        # 🔹 Kiểm tra collection chính
        COLLECTION_NAME = "tthc_vectors"
        if COLLECTION_NAME not in collections:
            print(f"❌ Collection '{COLLECTION_NAME}' không tồn tại!")
            return

        # 🔹 Lấy thông tin collection
        collection = Collection(COLLECTION_NAME)
        collection.load()

        # 🔹 Kiểm tra schema
        print("\n📋 Schema Collection:")
        print(collection.schema)

        # 🔹 Kiểm tra số lượng bản ghi
        num_entities = collection.num_entities
        print(f"\n📊 Số lượng bản ghi: {num_entities}")

        # 🔹 Câu hỏi từ người dùng
        test_query = "Các bước thực hiện thủ tục đăng ký khai tử?"
        print(f"\n🔍 Thử nghiệm tìm kiếm với câu hỏi: '{test_query}'")

        # Phân tích câu hỏi
        field, procedure_name, request = analyze_question(test_query)
        print(f"🔍 Lĩnh vực: {field}")
        print(f"🔍 Tên thủ tục: {procedure_name}")
        print(f"🔍 Yêu cầu: {request}")

        # Tạo embedding cho câu hỏi
        query_embedding = embedding_model.embed_query(test_query)

        # Thực hiện tìm kiếm vector từ câu hỏi
        search_params = {"metric_type": "L2", "params": {"nprobe": 20}}  # Tăng nprobe để cải thiện độ chính xác
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=5,  # Tăng số lượng kết quả trả về
            output_fields=["content", "linh_vuc", "section_name", "ten_thu_tuc"]
        )

        # Kiểm tra cấu trúc của kết quả tìm kiếm và trích xuất thông tin
        relevant_docs = []
        for hit in results:
            content = hit["content"] if "content" in hit else "Không có nội dung"
            
            # Lọc kết quả tìm kiếm chỉ lấy những tài liệu có liên quan đến lĩnh vực, thủ tục và yêu cầu
            if field.lower() in content.lower() and procedure_name.lower() in content.lower() and request.lower() in content.lower():
                relevant_docs.append(content)

        # Nếu có tài liệu phù hợp, dùng mô hình AI để phân tích và trả lời câu hỏi
        if relevant_docs:
            print("\n🤖 Tổng hợp câu trả lời từ LLM:")
            prompt = f"""
            Dưới đây là danh sách các tài liệu liên quan đến thủ tục hành chính tại Việt Nam.

            Tài liệu:
            {'\n\n'.join(relevant_docs)}

            Từ những tài liệu trên, hãy xác định thủ tục hành chính này thuộc lĩnh vực nào, tên thủ tục là gì và yêu cầu liên quan đến thủ tục này.
            Trả lời chi tiết và đúng các yêu cầu của câu hỏi, sử dụng thông tin từ tài liệu đã cho.
            """
            try:
                # Gọi mô hình AI để tổng hợp câu trả lời
                response = chat_model.invoke(prompt)
                print("\n✅ Câu trả lời:")
                print(response)
            except Exception as e:
                print(f"\n❌ Lỗi khi gọi LLM: {str(e)}")
        else:
            print("\n⚠️ Không tìm thấy tài liệu liên quan để trả lời câu hỏi.")

    except Exception as e:
        logger.error(f"Lỗi hệ thống: {str(e)}")
    finally:
        # Đóng kết nối
        try:
            collection.release()
            connections.disconnect("default")
            print("\n👋 Đã đóng kết nối!")
        except:
            pass

if __name__ == "__main__":
    check_system()
    