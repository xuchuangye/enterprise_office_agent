FROM python:3.10-slim

WORKDIR /app

# 【核心修复 1】安装 Linux 底层系统依赖
# libgomp1: PyTorch/OpenMP 必需，解决动态链接库初始化失败
# libgl1, libglib2.0-0: PyMuPDF 和某些视觉库必需
# build-essential: 编译某些 C 扩展包必需
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 【核心修复 2】强制使用 PyTorch CPU 索引源，防止拉取不兼容的 GPU 版本
# 这里通过环境变量将 PyTorch 的 CPU 源注入到 pip 中
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

RUN mkdir -p data/knowledge_base data/documents data/visualizations data/chroma_db

EXPOSE 8000
EXPOSE 8501

# 启动命令可根据需要修改
CMD ["sh", "-c", "uvicorn interface.fastapi_app:app --host 0.0.0.0 --port 8000 & streamlit run interface/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]