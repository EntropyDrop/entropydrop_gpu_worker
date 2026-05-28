# 1. 使用预装了 PyTorch 2.2.2 和 CUDA 12.1 的官方运行时镜像
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 设置环境变量，防止 apt 提示交互
ENV DEBIAN_FRONTEND=noninteractive

# 2. 安装系统依赖工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    xz-utils \
    autossh \
    git \
    wget \
    curl \
    openssh-client \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# 3. 安装 Node.js v22 并全局安装 PM2
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm i -g pm2 \
    && rm -rf /var/lib/apt/lists/*

# 5. 提前设置工作目录，并复制 requirements.txt 以安装依赖 (最大化利用 Docker 缓存机制)
WORKDIR /root/entropydrop_gpu_worker
COPY requirements.txt /root/entropydrop_gpu_worker/requirements.txt

# 6. 安装 Python 依赖包 (置于复制所有代码之前，防止代码改动导致重装)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    modelscope \
    diffusers>=0.30.2 \
    accelerate>=0.33.0 \
    peft \
    sentencepiece \
    protobuf \
    huggingface_hub


# 7. 下载大模型权重，固化在镜像内 (置于复制代码之前，防止代码改动导致重新下载巨额模型)
# 下载 FLUX.2-klein-base-4B 模型 (由 ModelScope 加速下载)
RUN python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('black-forest-labs/FLUX.2-klein-base-4B')"

# 下载 Sking LoRA 模型 (预置 HF 镜像站并支持运行时传入 HF_TOKEN 进行鉴权)
ENV HF_ENDPOINT="https://hf-mirror.com"
RUN pip install -U "huggingface_hub[cli]"
RUN hf download EntropyDrop/Sking \
    --repo-type model \
    --local-dir /root/Sking \
    --max-workers 4

# 9. 拷贝当前目录下的所有项目代码 (到此步为止，以上所有步骤都将完美走 CACHED 缓存)
COPY . /root/entropydrop_gpu_worker

# 启动容器并进入 entrypoint.sh 执行动态 SSH 隧道挂载与 PM2 启动
# ENTRYPOINT ["/entrypoint.sh"]
