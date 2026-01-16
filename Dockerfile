FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Default command (HF Space compatible)
CMD ["python", "-m", "training.train_unsloth_trl_singleasset"]
