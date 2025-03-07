from pymilvus import Collection, connections, utility
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv
import os
import random

# Tải các biến môi trường từ tệp .env
load_dotenv()

# Khởi tạo các biến môi trường từ tệp .env
ZILLIZ_URI = f"https://{os.getenv('ZILLIZ_HOST')}"
ZILLIZ_TOKEN = os.getenv('ZILLIZ_API_KEY')

# Kết nối tới Zilliz
def get_random_results_from_zilliz():
    try:
        # Kết nối tới Zilliz
        print("\U0001F50D Kết nối Zilliz tại", ZILLIZ_URI)
        connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

        # Chọn collection cần kiểm tra
        collection = Collection("tthc_vectors")
        
        # Lấy thông tin về collection
        print(f"Tổng số entities trong collection: {collection.num_entities}")

        # Hiển thị schema của collection
        print("\U0001F4CB Schema của collection 'tthc_vectors':")
        schema = collection.schema

        # In ra các trường trong schema
        print("Các trường dữ liệu trong schema:")
        for field in schema.fields:
            print(f"Field Name: {field.name}, Type: {field.dtype}")

        # Phương pháp 1: Lấy ngẫu nhiên bằng cách tạo các ID ngẫu nhiên
        # Lấy tổng số entities trong collection
        total_entities = collection.num_entities
        
        # Lấy danh sách các phần mục khác nhau để đảm bảo đa dạng
        # Sử dụng truy vấn đơn giản để lấy tất cả các giá trị duy nhất của section_name
        expr = "section_name != ''"
        results = collection.query(
            expr=expr,
            output_fields=["section_name"],
            limit=100  # Giới hạn số lượng kết quả
        )
        
        # Lấy danh sách các section_name duy nhất
        section_names = list(set([r["section_name"] for r in results if "section_name" in r]))
        print(f"\nCó {len(section_names)} phần mục khác nhau: {section_names}\n")
        
        # Lấy ngẫu nhiên 5 phần mục (hoặc ít hơn nếu không đủ 5)
        selected_sections = random.sample(section_names, min(5, len(section_names)))
        
        # Tạo một danh sách để lưu kết quả
        all_results = []
        
        # Với mỗi phần mục, lấy một kết quả ngẫu nhiên
        for section in selected_sections:
            expr = f"section_name == '{section}'"
            section_results = collection.query(
                expr=expr,
                output_fields=["id", "content", "ten_thu_tuc", "linh_vuc", "section_name", "ma_thu_tuc", "cap_thuc_hien"],
                limit=100  # Lấy tối đa 100 kết quả từ mỗi phần mục
            )
            
            if section_results:
                # Chọn ngẫu nhiên 1 kết quả từ mỗi phần mục
                random_result = random.choice(section_results)
                all_results.append(random_result)
        
        # Nếu chưa đủ 5 kết quả, bổ sung thêm bằng cách lấy ngẫu nhiên từ toàn bộ collection
        if len(all_results) < 5:
            # Lấy thêm kết quả ngẫu nhiên từ toàn bộ collection
            expr = ""  # Truy vấn rỗng để lấy tất cả
            additional_results = collection.query(
                expr=expr,
                output_fields=["id", "content", "ten_thu_tuc", "linh_vuc", "section_name", "ma_thu_tuc", "cap_thuc_hien"],
                limit=200  # Lấy một số lượng lớn để có thể chọn ngẫu nhiên
            )
            
            # Trộn kết quả
            random.shuffle(additional_results)
            
            # Thêm kết quả vào danh sách cho đến khi đủ 5
            for result in additional_results:
                if len(all_results) >= 5:
                    break
                    
                # Kiểm tra xem kết quả này đã có trong all_results chưa
                if not any(r["id"] == result["id"] for r in all_results):
                    all_results.append(result)
        
        # Hiển thị kết quả
        print("\n\U0001F50D Hiển thị 5 kết quả thực sự ngẫu nhiên từ Zilliz:")
        for i, result in enumerate(all_results[:5]):
            print(f"--- Kết quả #{i+1} ---")
            print(f"ID: {result['id']}")
            
            # Hiển thị các trường dữ liệu
            ten_thu_tuc = result.get("ten_thu_tuc", "Không có thông tin")
            ma_thu_tuc = result.get("ma_thu_tuc", "Không có thông tin")
            cap_thuc_hien = result.get("cap_thuc_hien", "Không có thông tin")
            linh_vuc = result.get("linh_vuc", "Không có thông tin")
            section_name = result.get("section_name", "Không có thông tin")
            content = result.get("content", "Không có nội dung")
            
            print(f"Tên thủ tục: {ten_thu_tuc}")
            print(f"Mã thủ tục: {ma_thu_tuc}")
            print(f"Cấp thực hiện: {cap_thuc_hien}")
            print(f"Lĩnh vực: {linh_vuc}")
            print(f"Phần mục: {section_name}")
            
            # Hiển thị toàn bộ nội dung
            print("\n--- NỘI DUNG ĐẦY ĐỦ ---")
            print(content)
            print("--- KẾT THÚC NỘI DUNG ---\n")
            
            print("="*50)

        # Ngắt kết nối
        connections.disconnect(alias="default")

    except Exception as e:
        print("❌ Lỗi kết nối hoặc truy vấn:", e)

# Kiểm tra phiên bản pymilvus
try:
    import pymilvus
    print(f"Phiên bản pymilvus: {pymilvus.__version__}")
except ImportError:
    print("Không thể import pymilvus để kiểm tra phiên bản")

# Chạy thử nghiệm để lấy kết quả ngẫu nhiên thực sự
get_random_results_from_zilliz()
