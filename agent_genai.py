"""
Telaiv LiveKit Agent — google-genai SDK 直接実装
livekit-plugins-google の mediaChunks 問題を回避するため
google.genai.live を直接使って Gemini Live に接続する
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from google import genai
from google.genai import types as genai_types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telaiv-agent-genai")

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 20  # 20ms チャンク

DEFAULT_GREETING = "はい、お電話ありがとうございます。"
DEFAULT_SYSTEM_PROMPT = """\
あなたは企業のAI受付担当です。
電話に出て、お客様の用件を丁寧に伺い、適切に対応してください。
常に丁寧な敬語を使い、用件を正確に聞き取ってメモを取ってください。
"""
DEFAULT_ESCALATION_KEYWORDS = ["クレーム", "担当者", "責任者", "上の者", "社長"]


# ─── Supabase ヘルパー ────────────────────────────────────────────────────────

async def get_concierge_config(called_number: str) -> Optional[dict]:
    try:
        pn = supabase.from_("phone_numbers") \
            .select("id, tenant_id") \
            .eq("number", called_number) \
            .eq("is_active", True) \
            .limit(1).execute()
        if not pn.data:
            return None
        phone_number_id = pn.data[0]["id"]
        tenant_id = pn.data[0]["tenant_id"]

        cc = supabase.from_("concierge_configs") \
            .select("*") \
            .eq("phone_number_id", phone_number_id) \
            .eq("is_enabled", True) \
            .limit(1).execute()
        if not cc.data:
            return {"tenant_id": tenant_id, "phone_number_id": phone_number_id}
        return cc.data[0]
    except Exception as e:
        logger.error(f"get_concierge_config error: {e}")
        return None


async def save_ai_conversation(
    config: Optional[dict],
    transcript: list[dict],
    outcome: str,
    duration_seconds: int,
    livekit_room_id: str,
) -> None:
    if not config or not config.get("tenant_id"):
        return
    try:
        supabase.from_("ai_conversations").insert({
            "tenant_id": config["tenant_id"],
            "transcript_json": transcript,
            "outcome": outcome,
            "gemini_model_used": GEMINI_MODEL,
            "livekit_room_id": livekit_room_id,
            "duration_seconds": duration_seconds,
        }).execute()
    except Exception as e:
        logger.error(f"save_ai_conversation error: {e}")


# ─── メイン会話ループ ─────────────────────────────────────────────────────────

async def run_conversation(
    ctx: JobContext,
    config: Optional[dict],
    audio_source: rtc.AudioSource,
    audio_track: rtc.RemoteAudioTrack,
) -> tuple[list[dict], str]:
    """Gemini Live と音声を双方向にブリッジする"""

    company = (config or {}).get("company_name") or "弊社"
    greeting_tmpl = (config or {}).get("greeting_template") or DEFAULT_GREETING
    greeting = greeting_tmpl.replace("{company}", company)
    system_prompt = (config or {}).get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    escalation_kw = (config or {}).get("escalation_keywords") or DEFAULT_ESCALATION_KEYWORDS

    full_instructions = f"{system_prompt}\n会社名: {company}"

    transcript: list[dict] = []
    escalated = False

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    live_config = genai_types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=full_instructions,
        speech_config=genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=(config or {}).get("voice") or "Puck"
                )
            )
        ),
    )

    audio_stream = rtc.AudioStream(audio_track, sample_rate=SAMPLE_RATE, num_channels=CHANNELS)

    async with client.aio.live.connect(model=GEMINI_MODEL, config=live_config) as session:
        logger.info("Gemini Live session established")

        # グリーティング送信
        await session.send(input=greeting, end_of_turn=True)

        async def recv_from_gemini():
            """Gemini の音声応答を LiveKit ルームに流す"""
            nonlocal escalated
            async for response in session.receive():
                if response.server_content:
                    for part in (response.server_content.model_turn.parts if response.server_content.model_turn else []):
                        if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                            frame = rtc.AudioFrame(
                                data=part.inline_data.data,
                                sample_rate=SAMPLE_RATE,
                                num_channels=CHANNELS,
                                samples_per_channel=len(part.inline_data.data) // 2,
                            )
                            await audio_source.capture_frame(frame)
                        if part.text:
                            transcript.append({
                                "role": "agent",
                                "text": part.text,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            for kw in escalation_kw:
                                if kw in part.text:
                                    escalated = True

        async def send_to_gemini():
            """LiveKit ルームの音声を Gemini に送る"""
            async for frame_event in audio_stream:
                frame = frame_event.frame
                await session.send(
                    input=genai_types.LiveClientRealtimeInput(
                        audio=genai_types.Blob(
                            data=bytes(frame.data),
                            mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                        )
                    )
                )

        try:
            await asyncio.gather(recv_from_gemini(), send_to_gemini())
        except Exception as e:
            logger.info(f"Conversation ended: {e}")

    outcome = "escalated" if escalated else ("resolved" if transcript else "abandoned")
    return transcript, outcome


# ─── エントリーポイント ───────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    logger.info(f"Job started: room={ctx.room.name}")
    await ctx.connect()

    start_time = datetime.now(timezone.utc)

    # 着信番号の取得
    called_number: Optional[str] = None
    try:
        room_meta = json.loads(ctx.room.metadata or "{}")
        called_number = room_meta.get("called_number") or room_meta.get("to")
    except (json.JSONDecodeError, AttributeError):
        pass

    if not called_number:
        for p in ctx.room.remote_participants.values():
            attrs = p.attributes or {}
            called_number = (
                attrs.get("sip.to")
                or attrs.get("sip.trunkPhoneNumber")
                or attrs.get("called_number")
            )
            if called_number:
                break

    logger.info(f"called_number={called_number}")

    config = await get_concierge_config(called_number) if called_number else None

    # LiveKit 音声 I/O セットアップ
    audio_source = rtc.AudioSource(SAMPLE_RATE, CHANNELS)
    local_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
    await ctx.room.local_participant.publish_track(local_track)

    # SIP 参加者の音声トラックを待機
    audio_track: Optional[rtc.RemoteAudioTrack] = None
    for attempt in range(30):
        for p in ctx.room.remote_participants.values():
            for pub in p.track_publications.values():
                if pub.track and isinstance(pub.track, rtc.RemoteAudioTrack):
                    audio_track = pub.track
                    break
            if audio_track:
                break
        if audio_track:
            break
        await asyncio.sleep(0.5)

    if not audio_track:
        logger.error("No remote audio track found, exiting")
        return

    logger.info("Audio track acquired, starting conversation")

    transcript, outcome = await run_conversation(ctx, config, audio_source, audio_track)

    duration = int((datetime.now(timezone.utc) - start_time).total_seconds())
    await save_ai_conversation(
        config=config,
        transcript=transcript,
        outcome=outcome,
        duration_seconds=duration,
        livekit_room_id=ctx.room.name,
    )
    logger.info(f"Job completed: outcome={outcome} duration={duration}s")


def _asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    exc = context.get("exception")
    if isinstance(exc, KeyError) and str(exc).startswith("'TR_"):
        return
    loop.default_exception_handler(context)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_asyncio_exception_handler)
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="telaiv-agent",
        )
    )
