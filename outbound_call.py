from __future__ import annotations

import base64
import json
import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

TELNYX_API_BASE = "https://api.telnyx.com/v2"


async def make_alert_call(
    caller_number: str,
    summary: str,
    urgency: str = "medium",
    caller_name: str = "お客様",
    to_number: str | None = None,
) -> bool:
    """クレーム/緊急検知時によしきさんに自動発信する。

    発信フロー:
    1. Telnyx が to_number（なければ NOTIFY_SMS_TO）に発信
    2. 応答後 → TTS でクレーム概要を読み上げ + DTMF ギャザー
    3. 1 → お客様の番号に転送 / 2 またはタイムアウト → 終話
    """
    api_key = os.environ.get("TELNYX_API_KEY")
    connection_id = os.environ.get("TELNYX_APP_ID")
    from_number = os.environ.get("TELNYX_PHONE_NUMBER")
    # to_number 引数優先、なければ env fallback
    resolved_to = to_number or os.environ.get("NOTIFY_SMS_TO")
    app_url = os.environ.get("NEXT_PUBLIC_APP_URL", "https://www.telaiv.com").rstrip("/")

    if not api_key or not connection_id or not from_number or not resolved_to:
        logger.warning(
            "make_alert_call: missing config — "
            "TELNYX_API_KEY / TELNYX_APP_ID / TELNYX_PHONE_NUMBER / notify_sms"
        )
        return False

    # Webhook 側に渡すデータを client_state (base64 JSON) にエンコード
    client_state_data = {
        "caller_number": caller_number,
        "caller_name": caller_name,
        "summary": summary[:80],   # メッセージ長を制限
        "urgency": urgency,
    }
    client_state = base64.b64encode(
        json.dumps(client_state_data, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")

    payload = {
        "connection_id": connection_id,
        "to": resolved_to,
        "from": from_number,
        "webhook_url": f"{app_url}/api/telnyx/outbound-webhook",
        "client_state": client_state,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{TELNYX_API_BASE}/calls",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.json()
                if resp.status in (200, 201, 202):
                    cid = body.get("data", {}).get("call_control_id", "")
                    logger.info(
                        f"make_alert_call: initiated to={resolved_to} urgency={urgency} "
                        f"call_control_id={cid}"
                    )
                    return True
                else:
                    logger.error(
                        f"make_alert_call: FAILED status={resp.status} body={body}"
                    )
                    return False
    except Exception:
        logger.error("make_alert_call: exception", exc_info=True)
        return False
