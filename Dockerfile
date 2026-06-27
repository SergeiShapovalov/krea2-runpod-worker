FROM runpod/worker-comfyui:5.8.6-base

ENV PYTHONUNBUFFERED=1 \
    MODEL_REPO_ID=pakkonen/krea2-base-bundle \
    MODEL_CACHE_WAIT_SECONDS=900 \
    COMFY_LOG_LEVEL=INFO \
    COMFY_API_AVAILABLE_INTERVAL_MS=500

COPY prepare_models.py /prepare_models.py
COPY handler.py /handler.py
COPY krea_start.sh /krea_start.sh

RUN chmod +x /krea_start.sh

CMD ["/krea_start.sh"]
