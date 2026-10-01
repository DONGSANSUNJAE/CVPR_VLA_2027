FROM python:3.11.11-slim-bookworm@sha256:a8e0a3090316aed0b11037aac613aef32fb1747dcc1dcb5c0f6c727a0113a07f
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:/opt/venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock
COPY app /opt/pi05
COPY docker/entrypoint.py /opt/pi05/entrypoint.py
RUN mkdir -p /opt/pi05/runtime_pro /opt/pi05/output /opt/pi05/checkpoints \
    && ln -s /opt/venv /opt/pi05/runtime_pro/.venv \
    && mv /opt/pi05/manifests /opt/pi05/frozen_manifests \
    && ln -s output/manifests /opt/pi05/manifests \
    && ln -s output/logs /opt/pi05/logs
ENV PYTHONPATH="/opt/pi05:/opt/pi05/scripts:/opt/pi05/openpi_snapshot/src:/opt/pi05/openpi_snapshot/packages/openpi-client/src" \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
    PI05_TOKENIZER_PATH=/opt/pi05/tokenizer/paligemma_tokenizer.model \
    OPENPI_DATA_HOME=/opt/pi05/output/cache/openpi \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.90 XLA_PYTHON_CLIENT_PREALLOCATE=true
WORKDIR /opt/pi05
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "/opt/pi05/entrypoint.py"]
CMD ["train"]
