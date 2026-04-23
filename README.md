# Telaiv LiveKit Agent

Telnyx SIP着信 → LiveKit Room → Gemini Live リアルタイム音声AIブリッジ

## アーキテクチャ

```
電話発信者
    ↓ (PSTN)
Telnyx
    ↓ (SIP Trunk)
LiveKit SIP Service
    ↓ (Room dispatch)
This Agent (telaiv-agent)
    ↓ (Gemini Live API)
Gemini 2.0 Flash (リアルタイム音声)
```

## セットアップ

### 1. 環境変数

```bash
cp .env.example .env
# .env を編集して各値を設定
```

### 2. ローカル実行

```bash
pip install -r requirements.txt
python agent.py dev   # 開発モード（ホットリロード付き）
python agent.py start # 本番モード
```

### 3. Docker

```bash
docker build -t telaiv-agent .
docker run --env-file .env telaiv-agent
```

### 4. Railway へのデプロイ

```bash
railway login
railway init
railway up
```

Railway の環境変数に `.env` の値を設定してください。

## LiveKit SIP 設定

LiveKit Cloud Dashboard で SIP Trunk と Dispatch Rule を設定します。

### SIP Trunk（Telnyx → LiveKit）

1. LiveKit Cloud → SIP → Inbound Trunk を作成
2. Telnyx Portal → SIP Connections → LiveKit の SIP URI を登録
   - LiveKit SIP URI 例: `sip:xxxxxxxx.sip.livekit.cloud`

### Dispatch Rule

着信番号ごとにルームを作成し、エージェントをディスパッチします:

```json
{
  "name": "telaiv-dispatch",
  "type": "direct_dispatch",
  "agent_name": "telaiv-agent",
  "room_name_format": "call-{sip.trunkPhoneNumber}-{random}"
}
```

ルームメタデータに `called_number` を含めることでエージェントが
Supabase から正しい concierge_config を取得します。

## Telnyx 設定

1. Telnyx Portal → SIP Connections → 新規作成
2. Outbound Voice Profile → LiveKit の SIP URI を設定
3. 取得済み番号 → SIP Connection を紐付け

## 動作フロー

1. 電話着信 → Telnyx が SIP で LiveKit に転送
2. LiveKit がルームを作成してこのエージェントをディスパッチ
3. エージェントが Supabase から `concierge_configs` を取得
4. Gemini Live でリアルタイム音声会話
5. エスカレーションキーワード検出 → 担当者転送通知
6. 通話終了 → `ai_conversations` に会話ログを保存

## エスカレーション

`concierge_configs.escalation_keywords` に登録したワードが検出されると:
- エージェントが「担当者におつなぎします」と応答
- ルームから切断（Telnyx 側でフォールバック処理）
- `outcome = "escalated"` で保存

## 注意事項

- Vercel サーバーレスでは動作しない（長時間接続が必要）
- Railway/Render/VPS にデプロイすること
- `GOOGLE_API_KEY` は Gemini API キー（Vertex AI ではなく AI Studio のもの）
- LiveKit SIP は LiveKit Cloud の有料プランで利用可能
