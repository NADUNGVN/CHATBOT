import os
import json
import logging
from dotenv import load_dotenv
import together
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load biến môi trường từ .env
load_dotenv()

class VectorDBCreator:
    def __init__(self):
        # API Key cho Together AI (Embedding Model)
        self.api_key = os.getenv("TOGETHER_API_KEY_1")
        self.together_client = together.Together(api_key=self.api_key)

        # Kết nối Zilliz Cloud
        self.uri = f"https://{os.getenv('ZILLIZ_HOST')}"
        self.token = os.getenv("ZILLIZ_API_KEY")
        self.collection_name = "tthc_vectors_1"
        self.dim = 768  # Kích thước vector embedding

        # Đường dẫn thư mục chứa chunks
        self.chunks_dir = os.path.join(os.getcwd(), "experimental_2", "data", "chunks")

        self.connect_zilliz()
        self.create_collection()

    def connect_zilliz(self):
        """Kết nối đến Zilliz Cloud"""
        connections.connect(uri=self.uri, token=self.token)
        logger.info("✅ Kết nối Zilliz Cloud thành công!")

    def create_collection(self):
        """Tạo collection trong Zilliz nếu chưa tồn tại"""
        if utility.has_collection(self.collection_name):
            logger.info(f"✅ Collection '{self.collection_name}' đã tồn tại.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="section_name", dtype=DataType.VARCHAR, max_length=255)
        ]

        schema = CollectionSchema(fields)
        collection = Collection(self.collection_name, schema)
        collection.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "HNSW"})

        logger.info(f"✅ Tạo collection '{self.collection_name}' thành công!")

    def get_embedding(self, text: str):
        """Tạo embedding từ văn bản sử dụng Together AI"""
        response = self.together_client.embeddings.create(
            input=[text],
            model="togethercomputer/m2-bert-80M-32k-retrieval"
        )
        return response.data[0].embedding

    def insert_chunks(self):
        """Quét tất cả file chunks trong thư mục và chèn vào Zilliz"""
        collection = Collection(self.collection_name)
        collection.load()

        if not os.path.exists(self.chunks_dir):
            logger.error(f"❌ Thư mục {self.chunks_dir} không tồn tại!")
            return

        files = [f for f in os.listdir(self.chunks_dir) if f.endswith(".json")]
        if not files:
            logger.error(f"❌ Không tìm thấy file chunks trong {self.chunks_dir}!")
            return

        logger.info(f"📂 Tìm thấy {len(files)} file chunks, bắt đầu insert...")

        for file_name in files:
            file_path = os.path.join(self.chunks_dir, file_name)
            logger.info(f"📥 Đang xử lý file: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            entities = []
            for chunk in chunks:
                content = chunk["content"]
                metadata = chunk["metadata"]
                embedding = self.get_embedding(content)

                entity = {
                    "embedding": embedding,
                    "content": content,
                    "intent": metadata.get("intent", ""),
                    "section_name": metadata.get("section_name", "")
                }
                entities.append(entity)

            collection.insert(entities)
            logger.info(f"✅ Đã insert {len(entities)} chunks từ file {file_name} vào Zilliz!")

        collection.flush()
        logger.info("✅ Hoàn thành insert tất cả file vào Zilliz!")

def main():
    creator = VectorDBCreator()
    creator.insert_chunks()

if __name__ == "__main__":
    main()
