FROM python:3.11-slim

WORKDIR /app

# システム依存パッケージ（音声処理に必要）
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py agent_genai.py report.py sms.py outbound_call.py outbound_runner.py outbound_genai.py dial_server.py start.sh ./
RUN chmod +x start.sh

# silero VAD モデルを事前ダウンロード（コールドスタート対策）
RUN python -c "from livekit.plugins import silero; silero.VAD.load()" || true

CMD ["./start.sh"]
