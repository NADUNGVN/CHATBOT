from pymilvus import Collection, connections
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv
import os

# Tải các biến môi trường từ tệp .env
load_dotenv()

# Khởi tạo các biến môi trường từ tệp .env
ZILLIZ_URI = f"https://{os.getenv('ZILLIZ_HOST')}"
ZILLIZ_TOKEN = os.getenv('ZILLIZ_API_KEY')

# Khởi tạo embedding model
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY_2')
embedding_model = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-32k-retrieval",
    together_api_key=TOGETHER_API_KEY
)

# Kết nối tới Zilliz
def test_zilliz_connection():
    try:
        # Kết nối tới Zilliz
        print("🔍 Kết nối Zilliz tại", ZILLIZ_URI)
        connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

        # Chọn collection cần kiểm tra
        collection = Collection("tthc_vectors")

        # Hiển thị schema của collection
        print("📋 Schema của collection 'tthc_vectors':")
        schema = collection.schema

        # In ra các trường trong schema
        print("Các trường dữ liệu trong schema:")
        for field in schema.fields:
            print(f"Field Name: {field.name}, Type: {field.dtype}")

        # Thực hiện tìm kiếm trên toàn bộ collection để lấy 5 kết quả ngẫu nhiên
        print("\n🔍 Lấy 5 kết quả ngẫu nhiên từ Zilliz:")

        # Tạo một truy vấn thực tế (embedding) thay vì vector ngẫu nhiên
        query = "thủ tục hành chính"
        query_embedding = embedding_model.embed_query(query)  # Lấy embedding từ mô hình

        # Kiểm tra kích thước của query_embedding
        print(f"Kích thước của query_embedding: {len(query_embedding)}")

        # Thực hiện tìm kiếm trên collection
        results = collection.search(
            data=[query_embedding], 
            anns_field="embedding",  # Trường lưu trữ vector nhúng
            param={"metric_type": "L2", "params": {"nprobe": 20}},
            limit=5,  # Lấy tối đa 5 kết quả
            output_fields=["id", "content", "ten_thu_tuc", "linh_vuc", "section_name"]  # Các trường cần lấy
        )

        # Hiển thị 5 kết quả ngẫu nhiên và kiểm tra cấu trúc của 'Hit'
        for i, result in enumerate(results[0]):  # Lấy kết quả từ search
            print(f"--- Kết quả #{i+1} ---")
            # Truy xuất các thuộc tính của 'Hit' đúng cách
            print(f"Kiểu của result: {type(result)}")  # Kiểu của đối tượng result

            # Truy xuất ID và khoảng cách từ đối tượng 'Hit'
            print(f"ID: {result.id}")
            print(f"Distance: {result.distance}")

            # Truy xuất các trường trong 'entity' từ 'Hit'
            entity = result.entity
            print(f"Tên thủ tục: {entity['ten_thu_tuc'] if 'ten_thu_tuc' in entity else 'Không có tên thủ tục'}")
            print(f"Content: {entity['content'][:100] if 'content' in entity else 'Không có nội dung'}...")  # In ra 100 ký tự đầu tiên của content
            print(f"Lĩnh vực: {entity['linh_vuc'] if 'linh_vuc' in entity else 'Không có lĩnh vực'}")
            print(f"Phần mục: {entity['section_name'] if 'section_name' in entity else 'Không có phần mục'}")
            print("="*50)

        # Ngắt kết nối
        connections.disconnect(alias="default")

    except Exception as e:
        print("❌ Lỗi kết nối hoặc truy vấn:", e)

# Chạy thử nghiệm kiểm tra schema và lấy 5 kết quả ngẫu nhiên
test_zilliz_connection()
