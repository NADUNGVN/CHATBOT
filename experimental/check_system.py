import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
from langchain_together import ChatTogether, TogetherEmbeddings
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

        # 🔹 Lấy mẫu dữ liệu
        print("\n📦 Lấy 3 bản ghi mẫu:")
        sample_query = collection.query(
            expr="",
            output_fields=[
                "id", "content", "ma_thu_tuc", "ten_thu_tuc",
                "cap_thuc_hien", "linh_vuc", "section_name"
            ],
            limit=3
        )

        for idx, result in enumerate(sample_query, 1):
            print(f"\n=== 📝 TÀI LIỆU {idx} ===")
            print(f"🔢 ID: {result.get('id', 'N/A')}")
            print(f"📄 Nội dung:\n{result.get('content', 'N/A')}")
            print(f"📌 Mã thủ tục: {result.get('ma_thu_tuc', 'N/A')}")
            print(f"📌 Tên thủ tục: {result.get('ten_thu_tuc', 'N/A')}")
            print(f"📌 Cấp thực hiện: {result.get('cap_thuc_hien', 'N/A')}")
            print(f"📌 Lĩnh vực: {result.get('linh_vuc', 'N/A')}")
            print(f"📌 Section: {result.get('section_name', 'N/A')}")
            print("=" * 50)

        # 🔹 Thử nghiệm tìm kiếm và trả lời
        test_query = "Tôi muốn đăng kí kết hôn?"
        print(f"\n🔍 Thử nghiệm tìm kiếm với câu hỏi: '{test_query}'")
        
        # Tạo embedding cho câu hỏi test
        query_embedding = embedding_model.embed_query(test_query)
        
        # Thực hiện tìm kiếm vector
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
            print(f"\nKết quả tìm kiếm: {hit}")

            # Truy xuất thông tin từ các trường trong đối tượng `hit`
            content = hit["content"] if "content" in hit else "Không có nội dung"
            linh_vuc = hit["linh_vuc"] if "linh_vuc" in hit else "N/A"
            section = hit["section_name"] if "section_name" in hit else "N/A"
            ten_thu_tuc = hit["ten_thu_tuc"] if "ten_thu_tuc" in hit else "N/A"
            
            # Kiểm tra độ tương đồng từ 'distance'
            distance = hit["distance"] if "distance" in hit else "Không có độ tương đồng"
            
            print(f"🎯 Độ tương đồng (hoặc điểm): {distance}")
            print(f"📄 Nội dung: {content}")
            print(f"📌 Lĩnh vực: {linh_vuc}")
            print(f"📌 Section: {section}")
            print(f"📌 Thủ tục: {ten_thu_tuc}")
            
            relevant_docs.append(content)  # Lưu các tài liệu liên quan

        # Sử dụng LLM để tổng hợp câu trả lời từ các tài liệu
        if relevant_docs:
            print("\n🤖 Tổng hợp câu trả lời từ LLM:")
            prompt = f"""
            Dựa trên các tài liệu sau, hãy trả lời câu hỏi: "{test_query}"

            Tài liệu:
            {'\n\n'.join(relevant_docs)}

            Hãy trả lời ngắn gọn và chỉ sử dụng thông tin từ tài liệu trên.
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
