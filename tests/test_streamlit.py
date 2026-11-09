"""
测试前端界面Streamlit
"""
import streamlit as st

# 设置标题
st.title("企业智能办公助手")
st.write("欢迎使用企业智能办公助手！")

# 文本输入框
name = st.text_input("请输入您的姓名：")

# 按钮
if st.button("打招呼"):
    st.write(f"你好，{name}！我是企业智能办公助手！")

# 启动Streamlit
# streamlit run test_streamlit.py
