# rag/index_manager.py
import os
import shutil
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.indices.loading import load_index_from_storage
from rag.config import DATA_DIR, INDEX_DIR

# 单例嵌入模型（核心：仅用HuggingFace全局缓存，不手动指定路径）
_EMBEDDING = None

def get_embedding():
    global _EMBEDDING
    if _EMBEDDING is not None:
        return _EMBEDDING

    print("🔧 加载嵌入模型（首次启动下载，后续直接缓存）...")
    # 极简初始化：仅传模型名+缓存（旧版本最兼容）
    _EMBEDDING = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        trust_remote_code=True,
        device="cpu"
    )
    print("✅ 嵌入模型加载完成！")
    return _EMBEDDING

# 加载文档
def load_documents() -> list:
    return SimpleDirectoryReader(DATA_DIR).load_data()

# 加载现有索引
def load_index():
    if not os.path.exists(INDEX_DIR):
        return None
    print("📌 正在加载现有向量库...")
    embedding = get_embedding()
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    return load_index_from_storage(storage_context, embed_model=embedding)

# 创建新索引
def create_index():
    docs = load_documents()
    if not docs:
        print("📭 未检测到任何文档，无法创建向量库")
        return None
    print(f"⚡ 基于 {len(docs)} 个文档创建新向量库...")
    embedding = get_embedding()
    index = VectorStoreIndex.from_documents(docs, embed_model=embedding)
    index.storage_context.persist(INDEX_DIR)
    print("✅ 向量库创建完成")
    return index

# 加载或创建索引
def load_or_create_index():
    index = load_index()
    return index if index is not None else create_index()

# 全量重建索引
def rebuild_index():
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        print("🗑️ 已删除旧向量库")
    return create_index()