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
    scenario: str = "test_intro"           # 後方互換
    stage: Optional[str] = None            # 新: seeding/watering/fertilizing/harvesting/inbound/test_intro
    customer_context: Optional[dict] = None  # 顧客マスター情報（Phase β で自動取得）
    tenant_context: Optional[dict] = None    # テナント企業情報
    allow_final_close: bool = False          # AI に最終契約権限を与えるか
    custom_overrides: Optional[dict] = None  # 追加指示


def _sb_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


async def fetch_stage_prompts_from_db(tenant_id: str) -> dict:
    """Fetch concierge_stage_prompts for a tenant from Supabase.

    Returns a dict mapping stage name -> prompt fields, e.g.:
      {"seeding": {"prompt_addition": "...", "opening_line": "...", ...}}
    Returns {} on error or if no prompts configured.
    """
    if not SUPABASE_URL or not SUPABASE_KEY or not tenant_id:
        return {}
    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: get concierge_config id for tenant
            async with session.get(
                f"{SUPABASE_URL}/rest/v1/concierge_configs",
                headers=_sb_headers(),
                params={
                    "tenant_id": f"eq.{tenant_id}",
                    "select": "id",
                    "order": "created_at.asc",
                    "limit": "1",
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if not resp.ok:
                    logger.warning(f"fetch_stage_prompts_from_db: concierge_configs query failed status={resp.status}")
                    return {}
                configs = await resp.json()
                if not configs:
                    return {}
                config_id = configs[0]["id"]

            # Step 2: get stage_prompts for that config
            async with session.get(
                f"{SUPABASE_URL}/rest/v1/concierge_stage_prompts",
                headers=_sb_headers(),
                params={
                    "concierge_config_id": f"eq.{config_id}",
                    "select": "stage,prompt_addition,opening_line,closing_line,max_duration_seconds,goals,forbidden_phrases",
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if not resp.ok:
                    logger.warning(f"fetch_stage_prompts_from_db: stage_prompts query failed status={resp.status}")
                    return {}
                rows = await resp.json()
                result = {r["stage"]: r for r in rows if r.get("stage")}
                logger.info(f"fetch_stage_prompts_from_db: tenant={tenant_id} stages={list(result.keys())}")
                return result
    except Exception:
        logger.warning("fetch_stage_prompts_from_db: exception", exc_info=True)
        return {}


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
    _stage = req.stage or req.scenario or "test_intro"
    _base_meta = {
        "type": "outbound_sales",
        "call_log_id": req.call_log_id,
        "scenario": req.scenario,   # 後方互換
        "stage": _stage,
        "caller_name": req.caller_name,
        "company_name": req.company_name,
        "tenant_id": req.tenant_id or "",
        "room_name": room_name,
        "allow_final_close": req.allow_final_close,
    }
    if req.customer_context:
        _base_meta["customer_context"] = req.customer_context
    if req.tenant_context:
        _base_meta["tenant_context"] = req.tenant_context
    elif req.company_name:
        _base_meta["tenant_context"] = {"company_name": req.company_name}
    if req.custom_overrides:
        _base_meta["custom_overrides"] = req.custom_overrides

    # Fetch DB stage prompts so agent can use tenant-customized prompts
    if req.tenant_id:
        db_stage_prompts = await fetch_stage_prompts_from_db(req.tenant_id)
        if db_stage_prompts:
            _base_meta["db_stage_prompts"] = db_stage_prompts

    room_metadata = json.dumps(_base_meta, ensure_ascii=False)

    logger.info(
        f"/outbound/dial: to={req.to_number} stage={_stage!r} "
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
            logger.info(f"[DEBUG] LIVEKIT_SIP_OUTBOUND_TRUNK_ID env value: {trunk_id!r}")

            # Agent を Room に明示 dispatch (SIP 発信より必ず先に実行)
            dispatch_metadata = room_metadata  # room と同一の metadata を使い回す
            logger.info(
                f"[DEBUG] About to call create_dispatch: agent_name=telaiv-agent room={room_name}"
            )
            dispatch = await lk.agent_dispatch.create_dispatch(
                lkapi.CreateAgentDispatchRequest(
                    agent_name="telaiv-agent",
                    room=room_name,
                    metadata=dispatch_metadata,
                )
            )
            logger.info(
                f"[DEBUG] agent dispatched: dispatch_id={dispatch.id} room={room_name}"
            )

            logger.info(
                f"[DEBUG] About to call create_sip_participant: "
                f"trunk_id={trunk_id!r} room={room_name!r} to={req.to_number!r}"
            )
            try:
                sip_participant = await lk.sip.create_sip_participant(
                    lkapi.CreateSIPParticipantRequest(
                        sip_trunk_id=trunk_id,
                        sip_call_to=req.to_number,
                        room_name=room_name,
                        participant_identity=f"callee-{req.call_log_id[:8]}",
                        participant_name=req.caller_name or req.to_number,
                        wait_until_answered=False,
                    )
                )
                logger.info(f"[DEBUG] create_sip_participant returned: {sip_participant}")
            except Exception as sip_exc:
                logger.exception(
                    f"[DEBUG] create_sip_participant raised: {type(sip_exc).__name__}: {sip_exc}"
                )
                raise
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
