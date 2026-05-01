"""
Telaiv Dial Server — HTTP endpoint for outbound call dispatch

FastAPI server that accepts dial requests from the Next.js app and
creates a LiveKit room + SIP participant via the outbound trunk.
agent_genai.py (scenario='test_intro') is auto-dispatched to the room.

Run: python dial_server.py  (listens on $PORT, default 8080)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("telaiv-dial-server")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

app = FastAPI(title="Telaiv Dial Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://telaiv.com",
        "https://www.telaiv.com",
        "https://telaiv.dev",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class DialRequest(BaseModel):
    to_number: str
    caller_name: str
    company_name: str
    call_log_id: str
    tenant_id: Optional[str] = None
    scenario: str = "test_intro"


def _sb_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


async def _mark_error(call_log_id: str, message: str) -> None:
    """outbound_call_logs を error 状態に更新する。"""
    if not SUPABASE_URL or not SUPABASE_KEY or not call_log_id:
        return
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{SUPABASE_URL}/rest/v1/outbound_call_logs"
            async with session.patch(
                url,
                headers=_sb_headers(),
                params={"id": f"eq.{call_log_id}"},
                json={"outcome": "error", "ai_summary": message[:200]},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if not resp.ok:
                    logger.error(f"_mark_error: patch failed status={resp.status}")
    except Exception:
        logger.error("_mark_error: exception", exc_info=True)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/outbound/dial")
async def outbound_dial(req: DialRequest):
    trunk_id = os.environ.get("LIVEKIT_SIP_OUTBOUND_TRUNK_ID")
    lk_url = os.environ.get("LIVEKIT_URL")
    lk_key = os.environ.get("LIVEKIT_API_KEY")
    lk_secret = os.environ.get("LIVEKIT_API_SECRET")

    if not trunk_id or not lk_url or not lk_key or not lk_secret:
        missing = [
            v for v, k in [
                (trunk_id, "LIVEKIT_SIP_OUTBOUND_TRUNK_ID"),
                (lk_url, "LIVEKIT_URL"),
                (lk_key, "LIVEKIT_API_KEY"),
                (lk_secret, "LIVEKIT_API_SECRET"),
            ] if not v
        ]
        msg = f"Missing LiveKit config: {missing}"
        logger.error(f"/outbound/dial: {msg}")
        await _mark_error(req.call_log_id, msg)
        raise HTTPException(status_code=500, detail=msg)

    room_name = f"outbound-{req.call_log_id}"
    room_metadata = json.dumps(
        {
            "type": "outbound_sales",
            "call_log_id": req.call_log_id,
            "scenario": req.scenario,
            "caller_name": req.caller_name,
            "company_name": req.company_name,
            "tenant_id": req.tenant_id or "",
            "room_name": room_name,
        },
        ensure_ascii=False,
    )

    logger.info(
        f"/outbound/dial: to={req.to_number} scenario={req.scenario} "
        f"room={room_name} call_log_id={req.call_log_id}"
    )

    try:
        from livekit import api as lkapi

        async with lkapi.LiveKitAPI(url=lk_url, api_key=lk_key, api_secret=lk_secret) as lk:
            await lk.room.create_room(
                lkapi.CreateRoomRequest(
                    name=room_name,
                    metadata=room_metadata,
                    empty_timeout=120,
                    departure_timeout=30,
                )
            )
            await lk.sip.create_sip_participant(
                lkapi.CreateSIPParticipantRequest(
                    sip_trunk_id=trunk_id,
                    sip_call_to=req.to_number,
                    room_name=room_name,
                    participant_identity=f"callee-{req.call_log_id[:8]}",
                    participant_name=req.caller_name or req.to_number,
                    wait_until_answered=False,
                )
            )
    except Exception as exc:
        msg = str(exc)
        logger.error(f"/outbound/dial: LiveKit error room={room_name}: {msg}", exc_info=True)
        await _mark_error(req.call_log_id, msg)
        raise HTTPException(status_code=500, detail=msg)

    logger.info(f"/outbound/dial: dispatched room={room_name}")
    return {"ok": True, "room": room_name}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
