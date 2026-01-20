import streamlit as st
import httpx
import json
import time

# --- 页面配置 ---
st.set_page_config(
    page_title="Memos RAG - 个人第二大脑",
    page_icon="🧠",
    layout="centered"
)

# 后端 API 地址
API_URL = "http://localhost:8000"

# --- 侧边栏 ---
with st.sidebar:
    st.title("🧠 Memos RAG")
    st.markdown("---")
    st.info("基于你的 Memos 笔记构建的智能问答系统。")
    
    if st.button("🔄 检查后端状态"):
        try:
            response = httpx.get(f"{API_URL}/")
            if response.status_code == 200:
                st.success("后端连接正常")
            else:
                st.error("后端异常")
        except:
            st.error("无法连接到后端")

    if st.button("⚡ 刷新知识库"):
        with st.spinner("正在同步 Memos 并重建索引..."):
            try:
                # 设置较长的超时时间，因为 embedding 可能需要几秒钟
                response = httpx.post(f"{API_URL}/refresh", timeout=60.0)
                if response.status_code == 200:
                    st.success("✅ " + response.json().get("message", "更新成功"))
                else:
                    st.error(f"❌ 更新失败: {response.status_code}")
            except Exception as e:
                st.error(f"❌ 请求出错: {e}")

    st.markdown("---")
    st.caption("Powered by LangChain + FastAPI + Streamlit")

# --- 主界面 ---
st.header("💬 我的第二大脑")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if prompt := st.chat_input("问问你的笔记..."):
    # 1. 显示用户提问
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端获取流式回答
    with st.chat_message("assistant"):
        placeholder = st.empty() # 用于动态更新内容
        full_response = ""
        
        try:
            # 使用 httpx 进行流式请求
            with httpx.stream("POST", f"{API_URL}/chat/stream", json={"question": prompt}, timeout=None) as response:
                if response.status_code != 200:
                    st.error(f"API 错误: {response.status_code}")
                else:
                    # 遍历 SSE 流
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            content = line[6:] # 去掉 "data: " 前缀
                            
                            if content == "[DONE]":
                                break
                            
                            full_response += content
                            # 模拟打字机效果，加上一个光标
                            placeholder.markdown(full_response + "▌")
            
            # 最终渲染（去掉光标）
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"发生错误: {e}")
