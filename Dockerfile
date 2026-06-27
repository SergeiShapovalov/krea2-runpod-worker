FROM runpod/worker-comfyui:5.8.6-base

ENV PYTHONUNBUFFERED=1 \
    MODEL_REPO_ID=pakkonen/krea2-base-bundle \
    MODEL_CACHE_WAIT_SECONDS=900 \
    COMFY_LOG_LEVEL=INFO \
    COMFY_API_AVAILABLE_INTERVAL_MS=500

COPY check_comfy_kitchen.py /tmp/check_comfy_kitchen.py

RUN cd /comfyui \
    && git fetch origin master \
    && git checkout -B master origin/master \
    && uv pip install -r /comfyui/requirements.txt \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do \
         [ -f "$r" ] && uv pip install -r "$r" || true; \
       done \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0" "comfy-kitchen==0.2.13" \
    && python /tmp/check_comfy_kitchen.py \
    && python /comfyui/main.py --quick-test-for-ci --cpu

COPY prepare_models.py /prepare_models.py
COPY handler.py /handler.py
COPY krea_start.sh /krea_start.sh

RUN chmod +x /krea_start.sh

CMD ["/krea_start.sh"]
