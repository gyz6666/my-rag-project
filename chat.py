# rag/chat.py
import sys
from rag.query_engine import get_query_engine
from rag.index_manager import rebuild_index, load_or_create_index

def main():
    print("📚 启动 DeepSeek 知识库对话系统...")
    
    try:
        # 初始化查询引擎
        query_engine = get_query_engine()
        if not query_engine:
            print("❌ 查询引擎初始化失败，程序退出")
            return
        
        print("="*50)
        print("💬 系统就绪！支持以下指令：")
        print("   → 输入问题：基于知识库生成回答")
        print("   → 输入 update：全量重建向量库")
        print("   → 输入 exit：退出对话系统")
        print("="*50)
        
        # 核心对话循环
        while True:
            user_input = input("\n请输入: ").strip()
            if not user_input:
                continue
            
            # 退出指令
            if user_input.lower() == "exit":
                print("👋 对话结束，感谢使用！")
                sys.exit(0)
            
            # 更新向量库指令
            if user_input.lower() == "update":
                print("🔄 开始全量重建向量库...")
                rebuild_index()
                # 重建后重新加载索引和查询引擎
                print("🔧 重新初始化查询引擎...")
                query_engine = get_query_engine()
                if not query_engine:
                    print("❌ 查询引擎重建失败，请检查文档后重试")
                else:
                    print("✅ 向量库及查询引擎更新完成！")
                continue
            
            # 正常问答逻辑
            try:
                print("🤔 正在检索知识库并生成回答...")
                response = query_engine.query(user_input)
                print("\n📝 回答：")
                print("-"*30)
                print(str(response))
                print("-"*30)
            except Exception as e:
                print(f"❌ 回答生成失败：{str(e)}")
                print("💡 建议检查：1. API密钥/BaseURL  2. 向量库完整性  3. 网络连接")
    
    # 全局异常捕获
    except KeyboardInterrupt:
        print("\n\n👋 用户中断操作，程序退出")
    except Exception as e:
        print(f"\n❌ 系统初始化失败：{str(e)}")
        print("💡 排查方向：")
        print("   1. 检查 config.py 中的 API_KEY/BASE_URL/MODEL 配置")
        print("   2. 确认 DATA_DIR 目录存在且有可读文档")
        print("   3. 验证 DeepSeek API 可正常访问")

if __name__ == "__main__":
    main()