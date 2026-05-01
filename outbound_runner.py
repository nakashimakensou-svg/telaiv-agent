"""
Telaiv Outbound Runner — AI営業発信ループ

起動: python outbound_runner.py
常駐プロセスとして動作し、active なキャンペーンを定期ポーリングして
スケジュール内のリードに対して Telnyx で発信する。

環境変数:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  TELNYX_API_KEY, TELNYX_APP_ID, TELNYX_PHONE_NUMBER (発信元番号フォールバック)
  NEXT_PUBLIC_APP_URL
  RUNNER_POLL_INTERVAL  ポーリング間隔（秒）、デフォルト 30
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
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
TELNYX_API_BASE = "https://api.telnyx.com/v2"
APP_URL = os.environ.get("NEXT_PUBLIC_APP_URL", "https://www.telaiv.com").rstrip("/")
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
        return False

    start_str: str = campaign.get("call_hours_start") or "09:00"
    end_str: str = campaign.get("call_hours_end") or "18:00"
    try:
        sh, sm = map(int, start_str[:5].split(":"))
        eh, em = map(int, end_str[:5].split(":"))
        start_minutes = sh * 60 + sm
        end_minutes = eh * 60 + em
        now_minutes = now_jst.hour * 60 + now_jst.minute
        return start_minutes <= now_minutes < end_minutes
    except Exception:
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


# ── Telnyx dialing ───────────────────────────────────────────────────────────

async def _dial_lead(
    session: aiohttp.ClientSession,
    campaign: dict,
    lead: dict,
    tenant_id: str,
) -> bool:
    """
    Telnyx で lead.phone_number に発信する。
    client_state に campaign_id / lead_id / tenant_id を base64 JSON で埋め込む。
    成功したら True を返す。
    """
    api_key = os.environ.get("TELNYX_API_KEY")
    connection_id = os.environ.get("TELNYX_APP_ID")
    from_number = campaign.get("caller_number") or os.environ.get("TELNYX_PHONE_NUMBER")

    if not api_key or not connection_id or not from_number:
        logger.error("_dial_lead: missing Telnyx config")
        return False

    client_state_data = {
        "type": "outbound_sales",
        "campaign_id": campaign["id"],
        "lead_id": lead["id"],
        "tenant_id": tenant_id,
        "script": campaign.get("script", ""),
    }
    client_state = base64.b64encode(
        json.dumps(client_state_data, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")

    payload = {
        "connection_id": connection_id,
        "to": lead["phone_number"],
        "from": from_number,
        "webhook_url": f"{APP_URL}/api/telnyx/outbound-webhook",
        "client_state": client_state,
        "timeout_secs": 30,
    }

    try:
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
                call_control_id = body.get("data", {}).get("call_control_id", "")
                logger.info(
                    f"_dial_lead: dialing lead={lead['id']} to={lead['phone_number']} "
                    f"call_control_id={call_control_id}"
                )
                # 発信ログ記録
                await sb_post(session, "outbound_call_logs", {
                    "campaign_id": campaign["id"],
                    "lead_id": lead["id"],
                    "telnyx_call_id": call_control_id,
                    "called_at": datetime.now(timezone.utc).isoformat(),
                    "outcome": "dialing",
                    "cost_usd": 0,
                })
                # リードを calling 状態に
                await sb_patch(
                    session, "outbound_leads",
                    {"id": lead["id"]},
                    {
                        "status": "calling",
                        "last_called_at": datetime.now(timezone.utc).isoformat(),
                        "retry_count": lead.get("retry_count", 0) + 1,
                    },
                )
                # calls_made をインクリメント
                await sb_patch(
                    session, "outbound_campaigns",
                    {"id": campaign["id"]},
                    {"calls_made": (campaign.get("calls_made") or 0) + 1},
                )
                return True
            else:
                logger.error(f"_dial_lead: FAILED status={resp.status} body={body}")
                return False
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

    # ── 予算チェック ─────────────────────────────────────────────────────
    if budget_limit > 0 and budget_used >= budget_limit:
        logger.info(f"campaign={campaign_id}: budget exceeded → pausing")
        await sb_patch(session, "outbound_campaigns", {"id": campaign_id}, {"status": "budget_exceeded"})
        return

    # ── スケジュールチェック ──────────────────────────────────────────────
    if not _within_schedule(campaign):
        logger.debug(f"campaign={campaign_id}: outside schedule, skipping")
        return

    # ── クレジット残高チェック ────────────────────────────────────────────
    balance = await _get_tenant_credits(session, tenant_id)
    await _check_low_credit_alert(session, tenant_id, balance)
    if balance <= 0:
        logger.info(f"campaign={campaign_id}: no credits → pausing")
        await sb_patch(session, "outbound_campaigns", {"id": campaign_id}, {"status": "paused"})
        return

    # ── 現在の同時発信数チェック ──────────────────────────────────────────
    active_count = await _active_call_count(session, campaign_id)
    slots = concurrent_calls - active_count
    if slots <= 0:
        logger.debug(f"campaign={campaign_id}: concurrent slots full ({active_count}/{concurrent_calls})")
        return

    # ── 発信対象リードを取得 ─────────────────────────────────────────────
    # pending か no_answer（retry_count < max_retries）のリードを slots 件取得
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

    leads_to_call = (pending_rows + retry_rows)[:slots]

    if not leads_to_call:
        # 発信対象がなければキャンペーン完了
        remaining = await sb_get(
            session,
            "outbound_leads",
            {
                "campaign_id": f"eq.{campaign_id}",
                "status": "in.(pending,no_answer,calling)",
                "select": "id",
            },
        )
        if not remaining:
            logger.info(f"campaign={campaign_id}: all leads processed → completed")
            await sb_patch(session, "outbound_campaigns", {"id": campaign_id}, {"status": "completed"})
        return

    # ── 発信 ─────────────────────────────────────────────────────────────
    for lead in leads_to_call:
        success = await _dial_lead(session, campaign, lead, tenant_id)
        if not success:
            logger.warning(f"campaign={campaign_id}: dial failed for lead={lead['id']}")


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
            "select": "id,retry_count,campaign_id",
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
