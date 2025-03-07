import os
from dotenv import load_dotenv
import together
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
import json
import logging
import glob
from typing import Optional, List, Dict

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBCreator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # API Key cho Together AI
        self.api_key = os.getenv("TOGETHER_API_KEY_1")
        self.together_client = together.Together(api_key=self.api_key)
        
        # Kết nối Zilliz
        self.uri = f"https://{os.getenv('ZILLIZ_HOST')}"
        self.token = os.getenv("ZILLIZ_API_KEY")
        self.connect_zilliz()
        
        # Cấu hình collection
        self.dim = 768  # Kích thước vector của m2-bert-80M-32k-retrieval
        self.collection_name = "tthc_vectors"

    def connect_zilliz(self):
        """Kết nối đến Zilliz Cloud"""
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token
            )
            logger.info("Kết nối Zilliz Cloud thành công")
        except Exception as e:
            logger.error(f"Lỗi kết nối Zilliz: {str(e)}")
            raise

    def create_collection(self):
        """Tạo collection trong Zilliz với partition key là lĩnh_vực"""
        collection_name = self.collection_name
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            logger.info("Collection đã tồn tại, sử dụng collection hiện có")
            return collection

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="ma_thu_tuc", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="ten_thu_tuc", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="cap_thuc_hien", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="linh_vuc", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="section_name", dtype=DataType.VARCHAR, max_length=255)
        ]
        
        schema = CollectionSchema(fields)
        collection = Collection(collection_name, schema)
        
        # Tạo index cho vector embedding
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        logger.info(f"Đã tạo collection {collection_name} thành công!")
        return collection

    def get_embedding(self, text: str) -> List[float]:
        """Tạo embedding từ text sử dụng Together AI"""
        try:
            text = text.strip().replace('\n', ' ')
            response = self.together_client.embeddings.create(
                input=[text],
                model="togethercomputer/m2-bert-80M-32k-retrieval"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding: {str(e)}")
            raise

    def process_chunks(self, chunks: List[Dict]):
        """Xử lý và insert chunks vào Zilliz"""
        collection = Collection(self.collection_name)
        
        try:
            collection.load()
            entities = []
            total_chunks = len(chunks)
            
            # Nhóm chunks theo lĩnh vực
            chunks_by_field = {}
            for chunk in chunks:
                linh_vuc = chunk.get('metadata', {}).get('linh_vuc', 'Không xác định')
                if linh_vuc not in chunks_by_field:
                    chunks_by_field[linh_vuc] = []
                chunks_by_field[linh_vuc].append(chunk)
            
            # Xử lý từng nhóm lĩnh vực
            for linh_vuc, field_chunks in chunks_by_field.items():
                logger.info(f"Đang xử lý lĩnh vực: {linh_vuc} ({len(field_chunks)} chunks)")
                
                for idx, chunk in enumerate(field_chunks, 1):
                    try:
                        metadata = chunk.get('metadata', {})
                        content = chunk.get('content', '')
                        
                        # Tạo embedding
                        embedding = self.get_embedding(content)
                        
                        entity = {
                            "embedding": embedding,
                            "content": content,
                            "ma_thu_tuc": metadata.get('ma_thu_tuc', ''),
                            "ten_thu_tuc": metadata.get('ten_thu_tuc', ''),
                            "cap_thuc_hien": metadata.get('cap_thuc_hien', ''),
                            "linh_vuc": metadata.get('linh_vuc', ''),
                            "section_name": metadata.get('section_name', '')
                        }
                        entities.append(entity)
                        
                        if idx % 10 == 0:
                            logger.info(f"Đã xử lý {idx}/{len(field_chunks)} chunks trong lĩnh vực {linh_vuc}")
                    
                    except Exception as e:
                        logger.error(f"Lỗi khi xử lý chunk {idx} của lĩnh vực {linh_vuc}: {str(e)}")
                        continue

            # Insert theo batch
            if entities:
                batch_size = 100
                for i in range(0, len(entities), batch_size):
                    batch = entities[i:i + batch_size]
                    try:
                        collection.insert(batch)
                        logger.info(f"Đã insert batch {i//batch_size + 1}")
                    except Exception as e:
                        logger.error(f"Lỗi khi insert batch: {str(e)}")
                
                collection.flush()
                logger.info(f"Đã insert thành công {len(entities)} entities")

        except Exception as e:
            logger.error(f"Lỗi trong quá trình xử lý chunks: {str(e)}")
            raise
        finally:
            collection.release()

def main():
    try:
        creator = VectorDBCreator()
        creator.create_collection()
        
        # Đọc tất cả các file chunks trong thư mục data/chunks
        chunks_dir = "data/chunks"
        chunks_files = glob.glob(os.path.join(chunks_dir, "*.json"))
        
        if not chunks_files:
            logger.error(f"Không tìm thấy file chunks nào trong thư mục {chunks_dir}")
            return
        
        for chunks_file in chunks_files:
            try:
                logger.info(f"Đang xử lý file: {chunks_file}")
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                creator.process_chunks(chunks)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý file {chunks_file}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi: {str(e)}")
    finally:
        connections.disconnect("default")
        logger.info("Đã đóng kết nối Zilliz Cloud")

if __name__ == "__main__":
    main()
