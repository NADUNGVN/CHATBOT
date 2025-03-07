import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
from langchain_together import ChatTogether, TogetherEmbeddings

# 🔹 Load API keys từ .env
load_dotenv()
ZILLIZ_URI = f"https://{os.getenv('ZILLIZ_HOST')}"
ZILLIZ_TOKEN = os.getenv("ZILLIZ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY_2")

# 🔹 Khởi tạo mô hình chat từ Together AI
chat_model = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    together_api_key=TOGETHER_API_KEY
)

# 🔹 Khởi tạo mô hình tạo embedding từ Together AI
embedding_model = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-32k-retrieval",
    together_api_key=TOGETHER_API_KEY
)

# 🔹 Kết nối đến Zilliz Cloud
print("🔄 Đang kết nối đến Zilliz Cloud...")
connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

# 🔹 Kiểm tra danh sách collections
collections = utility.list_collections()
if not collections:
    print("⚠️ Không có collection nào hoặc kết nối không thành công!")
    exit()
print(f"✅ Kết nối thành công! Collections hiện có: {collections}")

# 🔹 Collection cần kiểm tra
COLLECTION_NAME = "tthc_vectors_1"

if COLLECTION_NAME not in collections:
    print(f"❌ Collection '{COLLECTION_NAME}' không tồn tại!")
    exit()

# 🔹 Lấy collection
collection = Collection(COLLECTION_NAME)
collection.load()

# 🔹 Kiểm tra schema của collection
print("\n📌 **Schema Collection:**")
print(collection.schema)

# 🔹 Kiểm tra số lượng bản ghi trong collection
num_entities = collection.num_entities
print(f"\n📊 **Số lượng bản ghi trong collection:** {num_entities}")

# 🔹 Lấy một số mẫu dữ liệu từ collection
print("\n📦 **Lấy 5 bản ghi mẫu từ Zilliz:**")
sample_query = collection.query(
    expr="",
    output_fields=["id", "content", "intent", "section_name", "embedding"],
    limit=5
)

for idx, result in enumerate(sample_query, 1):
    print(f"\n=== 📝 **TÀI LIỆU {idx}** ===")
    print(f"🔢 **ID:** {result.get('id', 'Không có')}")
    print(f"📜 **Nội dung (content):**\n{result.get('content', 'Không có')}")
    print(f"📌 **Intent:** {result.get('intent', 'Không có')}")
    print(f"📌 **Section Name:** {result.get('section_name', 'Không có')}")
    embedding = result.get("embedding", None)
    if embedding:
        print(f"🔢 **Embedding:** {embedding[:10]}...")  # Chỉ in 10 giá trị đầu của embedding
    else:
        print("⚠️ **Embedding không có trong dữ liệu!**")
    print("=" * 50)

# 🔹 Kiểm tra API LLM của Together AI
def test_llm_api(query):
    """Hàm để kiểm tra API LLM với một truy vấn cụ thể."""
    try:
        response = chat_model.invoke(query)
        print("\n✅ **Kết quả từ LLM:**", response)
    except Exception as e:
        print(f"❌ **Lỗi khi gọi API LLM:** {str(e)}")

# 🔹 Tạo embedding với Together AI
def create_embedding(input_text):
    """Hàm để tạo embedding cho một đoạn văn bản."""
    try:
        embedding = embedding_model.embed_query(input_text)
        print("\n✅ **Embedding:**", embedding[:10])  # Chỉ in 10 giá trị đầu để kiểm tra
        return embedding
    except Exception as e:
        print(f"❌ **Lỗi khi tạo embedding:** {str(e)}")
        return None

def ask_llm_based_on_docs(query, documents):
    """Hỏi model nhưng chỉ cho phép nó trả lời dựa trên tài liệu"""
    
    prompt = f"""
    Dưới đây là các tài liệu liên quan đến câu hỏi của bạn:

    {documents}

    Chỉ sử dụng thông tin từ tài liệu trên để trả lời câu hỏi sau:
    "{query}"

    Nếu không tìm thấy câu trả lời trong tài liệu, hãy nói: "Tôi không tìm thấy thông tin liên quan trong tài liệu."
    """

    # 🔹 Gọi API của Together AI với prompt
    try:
        response = chat_model.invoke(prompt)
        return response
    except Exception as e:
        return f"❌ Lỗi khi gọi model: {str(e)}"

def search_in_zilliz(query_text):
    """Hàm tìm kiếm tài liệu tương tự bằng vector search trong Zilliz"""
    print(f"\n🔎 **Tìm kiếm tài liệu gần nhất với:** \"{query_text}\"")
    
    # 🔹 Lấy embedding của câu hỏi
    query_embedding = create_embedding(query_text)
    if query_embedding is None:
        print("❌ Không thể tạo embedding. Dừng tìm kiếm.")
        return

    # 🔹 Thực hiện tìm kiếm vector
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    search_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["id", "content", "intent", "section_name"]
    )

    # 🔹 Kiểm tra xem có kết quả tìm kiếm không
    if not search_results or len(search_results[0]) == 0:
        print("⚠️ Không tìm thấy tài liệu phù hợp. Model sẽ không trả lời.")
        return "Không tìm thấy tài liệu liên quan."

    # 🔹 Lấy nội dung tài liệu tìm thấy
    docs = []
    for idx, result in enumerate(search_results[0], 1):
        doc_content = result.entity.to_dict().get("content", "Không có nội dung")
        docs.append(doc_content)

    # 🔹 Kết hợp tài liệu vào prompt để gửi cho model
    combined_docs = "\n\n".join(docs)
    return ask_llm_based_on_docs(query_text, combined_docs)

if __name__ == "__main__":
    test_query = "Các bước thực hiện thủ tục đăng ký khai tử?"
    
    # 🔹 Thực hiện tìm kiếm và chỉ trả lời dựa trên tài liệu tìm thấy
    answer = search_in_zilliz(test_query)
    print("\n✅ **Câu trả lời dựa trên tài liệu:**", answer)

