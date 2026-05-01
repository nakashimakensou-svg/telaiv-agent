#!/bin/bash
set -e

# FastAPI dial server をバックグラウンドで起動
python dial_server.py &
DIAL_PID=$!

# LiveKit agent をバックグラウンドで起動
python agent_genai.py start &
AGENT_PID=$!

# 片方が落ちたらその終了コードを保持してもう一方も終了させる
wait -n $DIAL_PID $AGENT_PID
EXIT_CODE=$?
kill $DIAL_PID $AGENT_PID 2>/dev/null || true
exit $EXIT_CODE
