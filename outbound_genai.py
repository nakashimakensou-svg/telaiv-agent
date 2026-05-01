"""
Telaiv Outbound Genai Agent — AI営業発信エージェント

Telnyx 経由でリードに発信し、LiveKit ルームに接続後に Gemini Live で
AI営業会話を行う。通話終了後にクレジット消費・リード状態更新・Slack 通知を行う。

このファイルは agent_genai.py とは独立したエントリーポイントとして起動する:
  python outbound_genai.py start

LiveKit の entrypoint は "type": "outbound_sales" の client_state を持つルームにのみ反応する。
"""

from __future__ import annotations

import array
import asyncio
import base64
import io
import json
import logging
import os
import wave as _wave
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from google import genai
from google.genai import types as genai_types

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("telaiv-outbound-genai")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
COST_PER_MINUTE = 0.03

SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000

JST = timezone(timedelta(hours=9))

OUTBOUND_SYSTEM_PROMPT = """\
あなたは企業のAI営業担当です。リストのお客様に架電し、自社サービスに興味を持っていただくことが目的です。

【基本スタンス】
- 挨拶は簡潔に。相手の時間を尊重する。
- 押しつけがましくなく、あくまで提案ベースで話す。
- 相手が断ったら、無理に引き止めず丁重にお礼を言って終了する。
- 興味を示したら、次のステップ（資料送付・折り返し・アポ取得）に繋げる。

【会話の流れ】
1. 挨拶 + 自己紹介（会社名・名前）
2. 電話に出ていただいたお礼
3. スクリプトに沿った提案
4. 相手の反応を聞く
5. 興味あり → 次のステップへ
   興味なし / 断り → 丁重にお礼を言って終了
   不在・留守 → 終了

【禁止事項】
- しつこく食い下がらない
- 嘘や誇張をしない
- 5回以上往復したら丁重に終了する

【スクリプト】
{script}
"""


# ── audio utils ───────────────────────────────────────────────────────────────

def _resample_pcm16(data: bytes, in_rate: int, out_rate: int) -> bytes:
    if in_rate == out_rate or not data:
        return data
    arr = array.array("h", data)
    n_in = len(arr)
    n_out = int(n_in * out_rate / in_rate)
    out: array.array = array.array("h", [0] * n_out)
    for i in range(n_out):
        pos = i * in_rate / out_rate
        idx = int(pos)
        frac = pos - idx
        s0 = arr[idx]
        s1 = arr[idx + 1] if idx + 1 < n_in else s0
        out[i] = max(-32768, min(32767, int(s0 + frac * (s1 - s0))))
    return out.tobytes()


def _mix_pcm16(a: bytes, b: bytes) -> bytes:
    aa = array.array("h", a)
    ab = array.array("h", b)
    na, nb = len(aa), len(ab)
    n = max(na, nb)
    if na < n:
        aa.extend([0] * (n - na))
    if nb < n:
        ab.extend([0] * (n - nb))
    mixed = array.array("h", (max(-32768, min(32767, (x + y) >> 1)) for x, y in zip(aa, ab)))
    return mixed.tobytes()


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── Supabase helpers ───────────────────────────────────────────────────────────

def _get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _sb_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def _sb_patch(session: aiohttp.ClientSession, path: str, match: dict, data: dict) -> None:
    params = {k: f"eq.{v}" for k, v in match.items()}
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    async with session.patch(url, headers=_sb_headers(), params=params, json=data) as resp:
        if not resp.ok:
            body = await resp.text()
            logger.error(f"_sb_patch {path}: status={resp.status} body={body}")


async def _sb_post(session: aiohttp.ClientSession, path: str, data: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    async with session.post(url, headers=_sb_headers(), json=data) as resp:
        if not resp.ok:
            body = await resp.text()
            logger.error(f"_sb_post {path}: status={resp.status} body={body}")


async def _sb_get_one(session: aiohttp.ClientSession, path: str, match: dict) -> Optional[dict]:
    params = {k: f"eq.{v}" for k, v in match.items()}
    params["limit"] = "1"
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    async with session.get(url, headers=_sb_headers(), params=params) as resp:
        if resp.ok:
            rows = await resp.json()
            return rows[0] if rows else None
    return None


# ── Post-call processing ───────────────────────────────────────────────────────

async def _deduct_credits(
    session: aiohttp.ClientSession,
    tenant_id: str,
    duration_seconds: int,
) -> float:
    """通話時間に応じてクレジットを消費し、消費額を返す。"""
    cost = round(COST_PER_MINUTE * (duration_seconds / 60), 6)
    if cost <= 0:
        return 0.0
    tenant = await _sb_get_one(session, "tenants", {"id": tenant_id})
    if not tenant:
        return cost
    current = float(tenant.get("outbound_credits_usd") or 0)
    used = float(tenant.get("outbound_credits_used_usd") or 0)
    await _sb_patch(
        session, "tenants", {"id": tenant_id},
        {
            "outbound_credits_usd": max(0.0, round(current - cost, 6)),
            "outbound_credits_used_usd": round(used + cost, 6),
        },
    )
    return cost


async def _update_lead_outcome(
    session: aiohttp.ClientSession,
    lead_id: str,
    outcome: str,
    notes: str = "",
) -> None:
    STATUS_MAP = {
        "interested": "interested",
        "not_interested": "not_interested",
        "callback": "callback",
        "no_answer": "no_answer",
        "abandoned": "no_answer",
    }
    status = STATUS_MAP.get(outcome, "no_answer")
    await _sb_patch(
        session, "outbound_leads", {"id": lead_id},
        {"status": status, "notes": notes},
    )


async def _update_call_log(
    session: aiohttp.ClientSession,
    campaign_id: str,
    lead_id: str,
    call_control_id: str,
    duration_seconds: int,
    outcome: str,
    ai_summary: str,
    cost_usd: float,
) -> None:
    # telnyx_call_id で一致する最新レコードを更新
    url = f"{SUPABASE_URL}/rest/v1/outbound_call_logs"
    params = {
        "campaign_id": f"eq.{campaign_id}",
        "lead_id": f"eq.{lead_id}",
        "telnyx_call_id": f"eq.{call_control_id}",
        "order": "called_at.desc",
        "limit": "1",
    }
    async with session.patch(
        url,
        headers=_sb_headers(),
        params=params,
        json={
            "duration_seconds": duration_seconds,
            "outcome": outcome,
            "ai_summary": ai_summary,
            "cost_usd": cost_usd,
        },
    ) as resp:
        if not resp.ok:
            body = await resp.text()
            logger.error(f"_update_call_log: status={resp.status} body={body}")


async def _update_campaign_stats(
    session: aiohttp.ClientSession,
    campaign_id: str,
    connected: bool,
    interested: bool,
    cost_usd: float,
) -> None:
    campaign = await _sb_get_one(session, "outbound_campaigns", {"id": campaign_id})
    if not campaign:
        return
    updates: dict = {
        "budget_used_usd": round(float(campaign.get("budget_used_usd") or 0) + cost_usd, 6),
    }
    if connected:
        updates["calls_connected"] = int(campaign.get("calls_connected") or 0) + 1
    if interested:
        updates["leads_generated"] = int(campaign.get("leads_generated") or 0) + 1
    await _sb_patch(session, "outbound_campaigns", {"id": campaign_id}, updates)


async def _notify_slack_hot_lead(
    session: aiohttp.ClientSession,
    lead: dict,
    campaign_name: str,
    ai_summary: str,
) -> None:
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    phone = lead.get("phone_number", "不明")
    company = lead.get("company_name") or ""
    name = lead.get("contact_name") or ""
    text = (
        f"🎯 *ホットリード検出* — キャンペーン「{campaign_name}」\n"
        f"📞 {phone}  {company} {name}\n"
        f"📝 {ai_summary}"
    )
    try:
        async with session.post(
            webhook_url,
            json={"text": text},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as _:
            pass
    except Exception:
        logger.error("_notify_slack_hot_lead: failed", exc_info=True)


async def _analyze_conversation(transcript: list[dict], script: str) -> tuple[str, str]:
    """Claude API で会話を分析し (outcome, summary) を返す。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not transcript:
        return "no_answer", ""

    lines = "\n".join(
        f"{'AI' if t['role'] == 'ai' else '相手'}: {t['text']}"
        for t in transcript
    )
    prompt = (
        "以下はAI営業エージェントと顧客の通話記録です。\n\n"
        f"{lines}\n\n"
        "以下を JSON で返してください（他のテキスト不要）:\n"
        '{"outcome": "interested|not_interested|callback|no_answer", '
        '"summary": "50文字以内の要約"}'
    )

    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return data.get("outcome", "no_answer"), data.get("summary", "")
    except Exception:
        logger.error("_analyze_conversation: failed", exc_info=True)
    return "no_answer", ""


# ── Main conversation ─────────────────────────────────────────────────────────

async def run_outbound_conversation(ctx: JobContext, client_state: dict) -> None:
    campaign_id = client_state.get("campaign_id", "")
    lead_id = client_state.get("lead_id", "")
    tenant_id = client_state.get("tenant_id", "")
    script = client_state.get("script", "")
    call_control_id = client_state.get("call_control_id", "")

    logger.info(
        f"run_outbound_conversation: "
        f"campaign={campaign_id} lead={lead_id} tenant={tenant_id}"
    )

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    system_prompt = OUTBOUND_SYSTEM_PROMPT.format(script=script or "自社サービスの紹介")

    gemini_config = genai_types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=genai_types.Content(
            parts=[genai_types.Part(text=system_prompt)],
        ),
        speech_config=genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name="Zephyr"),
            )
        ),
        realtime_input_config=genai_types.RealtimeInputConfig(
            activity_handling=genai_types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            turn_coverage=genai_types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
        ),
        input_audio_transcription=genai_types.AudioTranscriptionConfig(),
        output_audio_transcription=genai_types.AudioTranscriptionConfig(),
    )

    caller_chunks: list[bytes] = []
    ai_chunks: list[bytes] = []
    transcript: list[dict] = []
    started_at = datetime.now(timezone.utc)
    connected = False

    await ctx.connect()
    room = ctx.room
    logger.info(f"run_outbound_conversation: connected to room={room.name}")

    async with gemini_client.aio.live.connect(
        model=GEMINI_MODEL, config=gemini_config
    ) as session:
        # 挨拶トリガー
        await asyncio.sleep(0.5)
        await session.send_client_content(
            genai_types.ClientContentParams(
                turns=[genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="[通話が繋がりました。挨拶してください。]")],
                )],
                turn_complete=True,
            )
        )
        connected = True

        async def listen_audio():
            async for event in room.on("track_subscribed"):
                track = event.track
                if track.kind != rtc.TrackKind.KIND_AUDIO:
                    continue
                audio_stream = rtc.AudioStream(track, sample_rate=SEND_SAMPLE_RATE, num_channels=1)
                async for frame_event in audio_stream:
                    frame = frame_event.frame
                    pcm = bytes(frame.data)
                    caller_chunks.append(pcm)
                    await session.send_realtime_input(
                        audio={"data": pcm, "mime_type": "audio/pcm;rate=16000"}
                    )

        async def receive_audio():
            source = rtc.AudioSource(RECV_SAMPLE_RATE, 1)
            track = rtc.LocalAudioTrack.create_audio_track("ai-audio", source)
            await room.local_participant.publish_track(track)

            while True:
                turn = session.receive()
                async for resp in turn:
                    if resp.data:
                        ai_chunks.append(resp.data)
                        frame = rtc.AudioFrame(
                            data=resp.data,
                            sample_rate=RECV_SAMPLE_RATE,
                            num_channels=1,
                            samples_per_channel=len(resp.data) // 2,
                        )
                        await source.capture_frame(frame)
                    if resp.server_content:
                        sc = resp.server_content
                        if sc.input_transcription and sc.input_transcription.text:
                            transcript.append({"role": "caller", "text": sc.input_transcription.text})
                        if sc.output_transcription and sc.output_transcription.text:
                            transcript.append({"role": "ai", "text": sc.output_transcription.text})
                        if sc.turn_complete:
                            break

        listen_task = asyncio.create_task(listen_audio())
        receive_task = asyncio.create_task(receive_audio())

        # 最大5分で強制終了
        try:
            await asyncio.wait_for(
                asyncio.gather(listen_task, receive_task, return_exceptions=True),
                timeout=300,
            )
        except asyncio.TimeoutError:
            logger.info("run_outbound_conversation: timeout after 5 min")
        finally:
            listen_task.cancel()
            receive_task.cancel()

    # ── 後処理 ────────────────────────────────────────────────────────────────
    ended_at = datetime.now(timezone.utc)
    duration_seconds = int((ended_at - started_at).total_seconds())

    outcome, summary = await _analyze_conversation(transcript, script)
    logger.info(f"run_outbound_conversation: outcome={outcome} summary={summary!r}")

    async with aiohttp.ClientSession() as http:
        cost_usd = await _deduct_credits(http, tenant_id, duration_seconds)

        await _update_lead_outcome(http, lead_id, outcome, summary)

        if call_control_id:
            await _update_call_log(
                http, campaign_id, lead_id, call_control_id,
                duration_seconds, outcome, summary, cost_usd,
            )

        is_interested = outcome == "interested"
        await _update_campaign_stats(http, campaign_id, connected, is_interested, cost_usd)

        if is_interested:
            campaign = await _sb_get_one(http, "outbound_campaigns", {"id": campaign_id})
            lead = await _sb_get_one(http, "outbound_leads", {"id": lead_id})
            if campaign and lead:
                await _notify_slack_hot_lead(http, lead, campaign.get("name", ""), summary)

    logger.info(
        f"run_outbound_conversation: done "
        f"duration={duration_seconds}s cost=${cost_usd:.4f} outcome={outcome}"
    )


# ── LiveKit entrypoint ────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    room_meta: str = ctx.room.metadata or "{}"
    try:
        meta = json.loads(room_meta)
    except json.JSONDecodeError:
        meta = {}

    # client_state は LiveKit → agent の job metadata に含まれる
    # Telnyx SIP → LiveKit の場合は room.sip_attributes か metadata に埋め込む
    client_state_b64 = meta.get("client_state", "")
    if not client_state_b64:
        logger.warning("entrypoint: no client_state in room metadata, skipping")
        return

    try:
        client_state = json.loads(base64.b64decode(client_state_b64).decode("utf-8"))
    except Exception:
        logger.error("entrypoint: failed to decode client_state", exc_info=True)
        return

    if client_state.get("type") != "outbound_sales":
        logger.info(f"entrypoint: type={client_state.get('type')} — not outbound_sales, skipping")
        return

    await run_outbound_conversation(ctx, client_state)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.environ.get("LIVEKIT_URL", ""),
            api_key=os.environ.get("LIVEKIT_API_KEY", ""),
            api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
        )
    )
