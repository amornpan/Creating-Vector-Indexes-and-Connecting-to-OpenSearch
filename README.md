# การสร้าง Hybrid Search ด้วย OpenSearch และ LlamaIndex

## ภาพรวมของบทเรียน

ในบทเรียนนี้เราจะเรียนรู้การสร้างระบบ Hybrid Search ที่รวมการค้นหาแบบ semantic search และ keyword search เข้าด้วยกัน โดยใช้:
- **OpenSearch** สำหรับ vector database และ keyword search
- **LlamaIndex** สำหรับการจัดการเอกสารและ indexing
- **BGE-M3** embedding model สำหรับการแปลงข้อความเป็น vector

---

## ส่วนที่ 1: การติดตั้ง Dependencies

```python
# ติดตั้ง LlamaIndex และ dependencies
!pip install llama-index -q
!pip install llama-index-embeddings-huggingface -q
!pip install llama-index-vector-stores-opensearch -q
!pip install requests -q
!pip install nest_asyncio -q
```

**คำอธิบาย:**
- `llama-index`: Framework หลักสำหรับการสร้าง RAG applications
- `llama-index-embeddings-huggingface`: สำหรับใช้ embedding models จาก Hugging Face
- `llama-index-vector-stores-opensearch`: connector สำหรับ OpenSearch
- `requests`: สำหรับการเรียก HTTP APIs
- `nest_asyncio`: แก้ปัญหา event loop ใน Jupyter Notebook

---

## ส่วนที่ 2: Import Modules และการตั้งค่าเริ่มต้น

```python
import os
import torch
import urllib.request
import pickle
import requests
import nest_asyncio
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.core.vector_stores.types import VectorStoreQueryMode

# Apply nest_asyncio to avoid runtime errors
nest_asyncio.apply()
```

**คำอธิบาย:**
- Import modules ที่จำเป็นสำหรับการทำงาน
- `nest_asyncio.apply()` แก้ปัญหา async ใน Jupyter environment

---

## ส่วนที่ 3: การกำหนดค่าคอนฟิก

```python
# กำหนดค่าสำหรับ OpenSearch
OPENSEARCH_ENDPOINT = "http://34.41.37.53:9200"
OPENSEARCH_INDEX = "aekanun_doc_index"  # ⚠️ เปลี่ยนเป็นชื่อของคุณ
TEXT_FIELD = "content"
EMBEDDING_FIELD = "embedding"

# Check if CUDA is available for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ใช้อุปกรณ์: {device}")
```

**คำอธิบาย:**
- กำหนด endpoint ของ OpenSearch cluster
- กำหนดชื่อ index (ต้องเป็นชื่อเฉพาะของแต่ละคน)
- ตรวจสอบว่ามี GPU ใช้งานได้หรือไม่

---

## ส่วนที่ 4: การสร้าง Hybrid Search Pipeline

```python
def create_hybrid_search_pipeline():
    pipeline_url = f"{OPENSEARCH_ENDPOINT}/_search/pipeline/hybrid-search-pipeline"
    headers = {'Content-Type': 'application/json'}

    pipeline_config = {
        "description": "Pipeline for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {
                        "technique": "min_max"
                    },
                    "combination": {
                        "technique": "harmonic_mean",
                        "parameters": {
                            "weights": [0.3, 0.7]  # keyword: 0.3, semantic: 0.7
                        }
                    }
                }
            }
        ]
    }

    try:
        response = requests.put(pipeline_url, headers=headers, data=json.dumps(pipeline_config))
        if response.status_code in [200, 201]:
            print(f"สร้าง hybrid search pipeline สำเร็จ: {response.json()}")
        else:
            print(f"ไม่สามารถสร้าง pipeline ได้: {response.text}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการสร้าง pipeline: {e}")
```

**คำอธิบาย:**
- สร้าง search pipeline ที่รวม keyword search และ semantic search
- ใช้ min-max normalization เพื่อปรับค่า scores ให้อยู่ในช่วงเดียวกัน
- ใช้ harmonic mean ในการรวม scores โดยให้น้ำหนัก keyword 30% และ semantic 70%

---

## ส่วนที่ 5: การดาวน์โหลดเอกสาร

```python
def download_corpus():
    os.makedirs('./corpus_input', exist_ok=True)
    urls = [
        ("https://storage.googleapis.com/llm-course/md/1.md", "./corpus_input/1.md"),
        ("https://storage.googleapis.com/llm-course/md/2.md", "./corpus_input/2.md"),
        ("https://storage.googleapis.com/llm-course/md/44.md", "./corpus_input/44.md"),
        ("https://storage.googleapis.com/llm-course/md/5555.md", "./corpus_input/5555.md")
    ]
    for url, path in urls:
        if not os.path.exists(path):
            print(f"กำลังดาวน์โหลด {url} ไปยัง {path}")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"ไม่สามารถดาวน์โหลด {url} ได้: {e}")

# ดาวน์โหลด corpus
print("กำลังดาวน์โหลดไฟล์...")
download_corpus()
```

**คำอธิบาย:**
- สร้างโฟลเดอร์สำหรับเก็บเอกสาร
- ดาวน์โหลดไฟล์ Markdown จาก Google Storage
- ตรวจสอบว่าไฟล์มีอยู่แล้วหรือไม่ก่อนดาวน์โหลด

---

## ส่วนที่ 6: การโหลดและแยกเอกสาร

```python
# โหลดเอกสาร Markdown จากไดเรกทอรี
reader = SimpleDirectoryReader(
    input_dir="./corpus_input",
    recursive=True,
    required_exts=[".md", ".markdown"]
)
documents = reader.load_data()
print(f"โหลดเอกสาร {len(documents)} ไฟล์สำเร็จ")

# สร้าง parser สำหรับ Markdown
md_parser = MarkdownNodeParser()
nodes = md_parser.get_nodes_from_documents(documents)
print(f"สร้าง {len(nodes)} nodes ด้วย MarkdownNodeParser สำเร็จ")
```

**คำอธิบาย:**
- `SimpleDirectoryReader`: อ่านไฟล์จากโฟลเดอร์
- `MarkdownNodeParser`: แยกเอกสาร Markdown เป็น chunks (nodes) ตาม structure
- การแยกเป็น nodes ช่วยให้ค้นหาได้แม่นยำและจัดการได้ง่าย

---

## ส่วนที่ 7: การตั้งค่า Embedding Model

```python
# ตั้งค่า embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device=device)
print(f"ตั้งค่าโมเดล embedding BAAI/bge-m3 สำเร็จ")

# ตรวจสอบขนาดของ embedding
embeddings = embed_model.get_text_embedding("test")
dim = len(embeddings)
print(f"ขนาด embedding: {dim}")
```

**คำอธิบาย:**
- **BGE-M3**: Multi-lingual embedding model ที่รองรับหลายภาษารวมทั้งภาษาไทย
- ทดสอบการทำงานและตรวจสอบขนาด vector (1024 dimensions)
- การทราบขนาด vector สำคัญสำหรับการตั้งค่า OpenSearch

---

## ส่วนที่ 8: การสร้าง OpenSearch Vector Store

```python
# ตั้งค่า OpensearchVectorClient
client = OpensearchVectorClient(
    endpoint=OPENSEARCH_ENDPOINT,
    index=OPENSEARCH_INDEX,
    dim=dim,
    embedding_field=EMBEDDING_FIELD,
    text_field=TEXT_FIELD,
    search_pipeline="hybrid-search-pipeline",
)
print(f"ตั้งค่า OpensearchVectorClient สำเร็จ สำหรับ index '{OPENSEARCH_INDEX}'")

# สร้าง vector store
vector_store = OpensearchVectorStore(client)

# สร้าง storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)
```

**คำอธิบาย:**
- สร้าง client สำหรับเชื่อมต่อกับ OpenSearch
- กำหนดให้ใช้ hybrid search pipeline ที่สร้างไว้
- `StorageContext`: กำหนดว่าจะเก็บ vectors ที่ไหน

---

## ส่วนที่ 9: การสร้าง Index และบันทึก

```python
# สร้าง index
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embed_model
)
print(f"สร้าง index สำเร็จ")

# บันทึก index ด้วย pickle
index_filename = f"{OPENSEARCH_INDEX}.pkl"
with open(index_filename, 'wb') as f:
    pickle.dump(index, f)
print(f"บันทึก index ลงในไฟล์ {index_filename} สำเร็จ")

print("เสร็จสิ้นกระบวนการทั้งหมด!")
```

**คำอธิบาย:**
- สร้าง `VectorStoreIndex` โดยส่ง nodes, storage context และ embedding model
- LlamaIndex จะแปลง nodes เป็น embeddings และเก็บใน OpenSearch
- บันทึก index object เพื่อใช้งานในภายหลัง

---

## สรุป: สิ่งที่เกิดขึ้นในระบบ

1. **Document Processing**: ไฟล์ Markdown → Documents → Nodes
2. **Embedding Creation**: Text → Vector embeddings (1024 dim)
3. **Storage**: เก็บทั้ง text และ vectors ใน OpenSearch
4. **Search Pipeline**: รวม keyword + semantic search แบบ weighted
5. **Index Creation**: สร้าง searchable index พร้อมใช้งาน

## ข้อดีของ Hybrid Search

- **Keyword Search**: ดีสำหรับคำค้นหาที่ต้องการความแม่นยำสูง
- **Semantic Search**: เข้าใจบริบทและความหมาย
- **Combined Results**: ได้ผลลัพธ์ที่ครอบคลุมและแม่นยำมากขึ้น

## การใช้งานต่อไป

หลังจากรันโค้ดนี้สำเร็จแล้ว คุณจะได้:
- OpenSearch index ที่พร้อมใช้งาน
- ไฟล์ `.pkl` สำหรับโหลด index ใหม่
- ระบบ hybrid search ที่พร้อมตอบคำถาม