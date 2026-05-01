"""
Telaiv Outbound Runner — AI営業発信ループ

起動: python outbound_runner.py
常駐プロセスとして動作し、active なキャンペーンを定期ポーリングして
スケジュール内のリードに対して LiveKit SIP Participant API で発信する。

発信フロー:
  1. LiveKit Room を作成（メタデータにキャンペーン/リード情報を格納）
  2. LiveKit SIP Outbound Trunk でリードの電話に発信
  3. outbound_genai.py エージェントが Room に自動参加して Gemini Live 会話

環境変数:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
  LIVEKIT_SIP_OUTBOUND_TRUNK_ID  ← LiveKit ダッシュボードで作成した SIP Outbound Trunk ID
  RUNNER_POLL_INTERVAL  ポーリング間隔（秒）、デフォルト 30
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("telaiv-outbound-runner")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
POLL_INTERVAL = int(os.environ.get("RUNNER_POLL_INTERVAL", "30"))
COST_PER_MINUTE = 0.03
LOW_CREDIT_THRESHOLD = 5.0

JST = timezone(timedelta(hours=9))


# ── Supabase helpers ──────────────────────────────────────────────────────────

def _sb_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def sb_get(session: aiohttp.ClientSession, path: str, params: dict | None = None):
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    async with session.get(url, headers=_sb_headers(), params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def sb_patch(session: aiohttp.ClientSession, path: str, match: dict, data: dict):
    params = {k: f"eq.{v}" for k, v in match.items()}
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    async with session.patch(url, headers=_sb_headers(), params=params, json=data) as resp:
        resp.raise_for_status()
        return await resp.json()


async def sb_post(session: aiohttp.ClientSession, path: str, data: dict):
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    async with session.post(url, headers=_sb_headers(), json=data) as resp:
        resp.raise_for_status()
        return await resp.json()


# ── Schedule check ────────────────────────────────────────────────────────────

def _within_schedule(campaign: dict) -> bool:
    """JST 現在時刻がキャンペーンの発信時間内かチェック。"""
    now_jst = datetime.now(JST)
    weekday = now_jst.isoweekday()  # Mon=1 … Sun=7

    call_days: list[int] = campaign.get("call_days") or [1, 2, 3, 4, 5]
    if weekday not in call_days:
        logger.info(
            f"_within_schedule: SKIP weekday={weekday} not in call_days={call_days} "
            f"now_jst={now_jst.strftime('%Y-%m-%d %H:%M')} (JST)"
        )
        return False

    # DB raw values をそのままログに出す（None なら fallback していることが分かる）
    raw_start = campaign.get("call_hours_start")
    raw_end = campaign.get("call_hours_end")
    logger.info(
        f"_within_schedule: DB raw call_hours_start={raw_start!r} call_hours_end={raw_end!r}"
    )

    # None / 空文字のときは「終日」をデフォルトにする
    start_str: str = raw_start or "00:00"
    end_str: str = raw_end or "23:59"

    try:
        sh, sm = map(int, start_str[:5].split(":"))
        eh, em = map(int, end_str[:5].split(":"))
        start_minutes = sh * 60 + sm
        end_minutes = eh * 60 + em
        now_minutes = now_jst.hour * 60 + now_jst.minute
        in_window = start_minutes <= now_minutes < end_minutes
        logger.info(
            f"_within_schedule: now_jst={now_jst.strftime('%H:%M')} "
            f"window={start_str[:5]}-{end_str[:5]} → {'OK' if in_window else 'SKIP'}"
        )
        return in_window
    except Exception:
        logger.error("_within_schedule: parse error", exc_info=True)
        return False


# ── Credit helpers ─────────────────────────────────────────────────────────────

async def _get_tenant_credits(session: aiohttp.ClientSession, tenant_id: str) -> float:
    rows = await sb_get(
        session,
        "tenants",
        {"id": f"eq.{tenant_id}", "select": "outbound_credits_usd,outbound_credits_used_usd"},
    )
    if not rows:
        return 0.0
    return float(rows[0].get("outbound_credits_usd") or 0)


async def _deduct_credit(session: aiohttp.ClientSession, tenant_id: str, cost: float) -> None:
    """クレジット残高を減算し used を加算する（非同期・ベストエフォート）。"""
    try:
        rows = await sb_get(
            session,
            "tenants",
            {"id": f"eq.{tenant_id}", "select": "outbound_credits_usd,outbound_credits_used_usd"},
        )
        if not rows:
            return
        current = float(rows[0].get("outbound_credits_usd") or 0)
        used = float(rows[0].get("outbound_credits_used_usd") or 0)
        await sb_patch(
            session,
            "tenants",
            {"id": tenant_id},
            {
                "outbound_credits_usd": max(0.0, round(current - cost, 6)),
                "outbound_credits_used_usd": round(used + cost, 6),
            },
        )
    except Exception:
        logger.error("_deduct_credit: failed", exc_info=True)


async def _check_low_credit_alert(
    session: aiohttp.ClientSession, tenant_id: str, balance: float
) -> None:
    """残高が LOW_CREDIT_THRESHOLD 未満ならSlack通知 + SMS。"""
    if balance >= LOW_CREDIT_THRESHOLD:
        return
    try:
        slack_url = os.environ.get("SLACK_WEBHOOK_URL")
        if slack_url:
            async with session.post(
                slack_url,
                json={"text": f"⚠️ テナント {tenant_id} の発信クレジット残高が低下しています: ${balance:.2f}"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as _:
                pass
    except Exception:
        pass


# ── LiveKit SIP dialing ───────────────────────────────────────────────────────

async def _dial_lead(
    session: aiohttp.ClientSession,
    campaign: dict,
    lead: dict,
    tenant_id: str,
) -> bool:
    """
    LiveKit Room を作成し SIP Outbound Trunk でリードに発信する。
    Room メタデータにキャンペーン/リード情報を格納する。
    outbound_genai.py エージェントが Room に自動参加して会話する。
    """
    trunk_id = os.environ.get("LIVEKIT_SIP_OUTBOUND_TRUNK_ID")
    lk_url = os.environ.get("LIVEKIT_URL")
    lk_key = os.environ.get("LIVEKIT_API_KEY")
    lk_secret = os.environ.get("LIVEKIT_API_SECRET")

    if not trunk_id or not lk_url or not lk_key or not lk_secret:
        logger.error(
            "_dial_lead: missing LiveKit config "
            "(LIVEKIT_SIP_OUTBOUND_TRUNK_ID / LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET)"
        )
        return False

    room_name = f"outbound-{campaign['id'][:8]}-{lead['id'][:8]}-{int(time.time())}"
    room_metadata = json.dumps(
        {
            "type": "outbound_sales",
            "campaign_id": campaign["id"],
            "lead_id": lead["id"],
            "tenant_id": tenant_id,
            "script": campaign.get("script", ""),
            "room_name": room_name,
        },
        ensure_ascii=False,
    )

    try:
        from livekit import api as lkapi

        async with lkapi.LiveKitAPI(url=lk_url, api_key=lk_key, api_secret=lk_secret) as lk:
            await lk.room.create_room(
                lkapi.CreateRoomRequest(
                    name=room_name,
                    metadata=room_metadata,
                    empty_timeout=60,
                    departure_timeout=30,
                )
            )
            await lk.sip.create_sip_participant(
                lkapi.CreateSIPParticipantRequest(
                    sip_trunk_id=trunk_id,
                    sip_call_to=lead["phone_number"],
                    room_name=room_name,
                    participant_identity=f"lead-{lead['id'][:8]}",
                    participant_name=lead.get("contact_name") or lead["phone_number"],
                    wait_until_answered=False,
                )
            )

        logger.info(
            f"_dial_lead: room={room_name} lead={lead['id']} to={lead['phone_number']}"
        )
        await sb_post(
            session,
            "outbound_call_logs",
            {
                "campaign_id": campaign["id"],
                "lead_id": lead["id"],
                "telnyx_call_id": room_name,
                "called_at": datetime.now(timezone.utc).isoformat(),
                "outcome": "dialing",
                "cost_usd": 0,
            },
        )
        await sb_patch(
            session,
            "outbound_leads",
            {"id": lead["id"]},
            {
                "status": "calling",
                "last_called_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": lead.get("retry_count", 0) + 1,
            },
        )
        return True
    except Exception:
        logger.error("_dial_lead: exception", exc_info=True)
        return False


# ── Active call count check ───────────────────────────────────────────────────

async def _active_call_count(session: aiohttp.ClientSession, campaign_id: str) -> int:
    """calling 状態のリード数（≒ 現在通話中の数）を返す。"""
    rows = await sb_get(
        session,
        "outbound_leads",
        {"campaign_id": f"eq.{campaign_id}", "status": "eq.calling", "select": "id"},
    )
    return len(rows)


# ── Campaign processing ────────────────────────────────────────────────────────

async def _process_campaign(session: aiohttp.ClientSession, campaign: dict) -> None:
    tenant_id = campaign["tenant_id"]
    campaign_id = campaign["id"]
    concurrent_calls = int(campaign.get("concurrent_calls") or 1)
    max_retries = int(campaign.get("max_retries") or 2)
    budget_limit = float(campaign.get("budget_limit_usd") or 0)
    budget_used = float(campaign.get("budget_used_usd") or 0)

    logger.info(
        f"campaign={campaign_id}: processing "
        f"budget={budget_used:.2f}/{budget_limit:.2f} concurrent_calls={concurrent_calls} max_retries={max_retries}"
    )

    # ── 予算チェック ─────────────────────────────────────────────────────
    if budget_limit > 0 and budget_used >= budget_limit:
        logger.info(f"campaign={campaign_id}: SKIP budget exceeded ({budget_used:.2f} >= {budget_limit:.2f}) → pausing")
        await sb_patch(session, "outbound_campaigns", {"id": campaign_id}, {"status": "budget_exceeded"})
        return

    # ── スケジュールチェック ──────────────────────────────────────────────
    if not _within_schedule(campaign):
        logger.info(f"campaign={campaign_id}: SKIP outside schedule")
        return

    # ── クレジット残高チェック ────────────────────────────────────────────
    balance = await _get_tenant_credits(session, tenant_id)
    logger.info(f"campaign={campaign_id}: tenant={tenant_id} credit_balance=${balance:.4f}")
    await _check_low_credit_alert(session, tenant_id, balance)
    if balance <= 0:
        logger.info(f"campaign={campaign_id}: SKIP no credits (balance=${balance:.4f}) → pausing")
        await sb_patch(session, "outbound_campaigns", {"id": campaign_id}, {"status": "paused"})
        return

    # ── 現在の同時発信数チェック ──────────────────────────────────────────
    active_count = await _active_call_count(session, campaign_id)
    slots = concurrent_calls - active_count
    logger.info(f"campaign={campaign_id}: active_calls={active_count} slots={slots}/{concurrent_calls}")
    if slots <= 0:
        logger.info(f"campaign={campaign_id}: SKIP concurrent slots full ({active_count}/{concurrent_calls})")
        return

    # ── 発信対象リードを取得 ─────────────────────────────────────────────
    pending_rows = await sb_get(
        session,
        "outbound_leads",
        {
            "campaign_id": f"eq.{campaign_id}",
            "status": "eq.pending",
            "select": "*",
            "limit": str(slots),
            "order": "created_at.asc",
        },
    )
    logger.info(f"campaign={campaign_id}: pending_leads={len(pending_rows)}")

    retry_rows: list[dict] = []
    if len(pending_rows) < slots:
        retry_rows = await sb_get(
            session,
            "outbound_leads",
            {
                "campaign_id": f"eq.{campaign_id}",
                "status": "eq.no_answer",
                "retry_count": f"lt.{max_retries}",
                "select": "*",
                "limit": str(slots - len(pending_rows)),
                "order": "last_called_at.asc.nullsfirst",
            },
        )
        logger.info(f"campaign={campaign_id}: retry_leads={len(retry_rows)}")

    leads_to_call = (pending_rows + retry_rows)[:slots]
    logger.info(f"campaign={campaign_id}: leads_to_call={len(leads_to_call)}")

    if not leads_to_call:
        # 発信対象がなければキャンペーン完了チェック
        remaining = await sb_get(
            session,
            "outbound_leads",
            {
                "campaign_id": f"eq.{campaign_id}",
                "status": "in.(pending,no_answer,calling)",
                "select": "id",
            },
        )
        logger.info(f"campaign={campaign_id}: no leads to call, remaining_active={len(remaining)}")
        if not remaining:
            logger.info(f"campaign={campaign_id}: all leads processed → completed")
            await sb_patch(session, "outbound_campaigns", {"id": campaign_id}, {"status": "completed"})
        return

    # ── 発信 ─────────────────────────────────────────────────────────────
    dialed_count = 0
    for lead in leads_to_call:
        logger.info(f"campaign={campaign_id}: dialing lead={lead['id']} phone={lead.get('phone_number')}")
        success = await _dial_lead(session, campaign, lead, tenant_id)
        if success:
            dialed_count += 1
        else:
            logger.warning(f"campaign={campaign_id}: dial failed for lead={lead['id']}")

    # calls_made をまとめて加算（ループ内のスナップショット加算を避ける）
    if dialed_count > 0:
        current = await sb_get(
            session,
            "outbound_campaigns",
            {"id": f"eq.{campaign_id}", "select": "calls_made"},
        )
        base = int((current[0].get("calls_made") or 0) if current else 0)
        await sb_patch(
            session, "outbound_campaigns",
            {"id": campaign_id},
            {"calls_made": base + dialed_count},
        )


# ── Stale call cleanup ────────────────────────────────────────────────────────

async def _cleanup_stale_calls(session: aiohttp.ClientSession) -> None:
    """calling 状態のまま 10 分以上経過したリードを no_answer にリセット。"""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    rows = await sb_get(
        session,
        "outbound_leads",
        {
            "status": "eq.calling",
            "last_called_at": f"lt.{cutoff}",
            "select": "id,campaign_id",
        },
    )
    for lead in rows:
        logger.info(f"_cleanup_stale_calls: resetting lead={lead['id']}")
        await sb_patch(
            session, "outbound_leads",
            {"id": lead["id"]},
            {"status": "no_answer"},
        )


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run_loop() -> None:
    logger.info(f"Outbound runner started (poll_interval={POLL_INTERVAL}s)")
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                await _cleanup_stale_calls(session)

                campaigns = await sb_get(
                    session,
                    "outbound_campaigns",
                    {"status": "eq.active", "select": "*"},
                )
                logger.info(f"Active campaigns: {len(campaigns)}")
                for campaign in campaigns:
                    await _process_campaign(session, campaign)
            except Exception:
                logger.error("run_loop: unhandled exception", exc_info=True)

            await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(run_loop())
