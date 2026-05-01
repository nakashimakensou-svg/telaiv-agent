#!/bin/bash
set -e

# FastAPI dial server をバックグラウンドで起動
python dial_server.py &
DIAL_PID=$!

# LiveKit agent をフォアグラウンドで起動
python agent_genai.py start &
AGENT_PID=$!

# どちらかが終了したらもう一方も終了
wait -n $DIAL_PID $AGENT_PID
kill $DIAL_PID $AGENT_PID 2>/dev/null || true
