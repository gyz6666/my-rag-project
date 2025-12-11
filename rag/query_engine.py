 # rag/query_engine.py
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings  # 新增：配置全局LLM
from rag.config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL
from rag.index_manager import load_or_create_index

def get_llm():
    """返回 DeepSeek 的 LLM 对象（适配 OpenAI 兼容接口）"""
    # 手动指定 DeepSeek 模型的上下文窗口大小
    context_window = 4096 if MODEL == "deepseek-chat" else 32768
    
    return OpenAILike(
        api_key=OPENAI_API_KEY,
        api_base=OPENAI_BASE_URL,  # 注意参数名是 api_base 而非 base_url
        model=MODEL,
        context_window=context_window,
        is_chat_model=True,
        temperature=0.7,
    )

# 【确保函数名拼写正确】
def get_query_engine():
    """返回 Query Engine 对象"""
    # 配置全局LLM（避免索引调用时找不到LLM）
    llm = get_llm()
    Settings.llm = llm
    
    # 加载/创建索引
    index = load_or_create_index()
    if index is None:
        raise ValueError("向量索引加载/创建失败，请检查DATA_DIR是否有文档")
    
    # 创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        llm=llm
    )

    return query_engine