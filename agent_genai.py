"""
Telaiv LiveKit Agent — google-genai SDK direct implementation
Based on official Google Gemini cookbook Live API examples.
https://github.com/google-gemini/cookbook/tree/main/quickstarts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from google import genai
from google.genai import types as genai_types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telaiv-agent-genai")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview")

# Gemini Live I/O sample rates (per official cookbook)
SEND_SAMPLE_RATE = 16000   # LiveKit → Gemini: 16kHz PCM
RECV_SAMPLE_RATE = 24000   # Gemini → LiveKit: 24kHz PCM
CHANNELS = 1

DEFAULT_GREETING = "はい、お電話ありがとうございます。"
DEFAULT_SYSTEM_PROMPT = """\
あなたは企業のAI電話受付担当です。人間のオペレーターのように自然な会話をしてください。

【基本姿勢】
- 常に丁寧な敬語を使い、落ち着いたトーンで話す
- 相手のペースに合わせ、決して急かさない
- 用件を正確に聞き取り、繰り返して確認する

【自然な会話の振る舞い】
- 相手の発言に対して「そうですか」「なるほど」「かしこまりました」など自然な相槌を入れる
- 相手が話し終わるまで遮らず、しっかり聞く
- 会話が途切れたり間が生じたら「いかがでしょうか？」「他にご用件はございますか？」など自然につなぐ

【状況対応】
- 相手が咳き込んだり体調が悪そうな場合は「大丈夫でしょうか？」「お気をつけください」と気遣う
- 相手が無言・返答がない場合は数秒待ってから「もしもし、聞こえていますか？」と穏やかに確認する
- 相手が怒っているときは落ち着いたトーンで「大変申し訳ございません」と誠実に対応する
- 相手が高齢者や話すのが遅い場合はゆっくり丁寧に、若い人には少し親しみやすいトーンで

【禁止事項】
- 同じ質問を繰り返さない
- 長々と説明せず、簡潔に要点を伝える
- 不明な点は正直に「確認いたします」と伝え、でたらめを言わない
"""
DEFAULT_ESCALATION_KEYWORDS = ["クレーム", "担当者", "責任者", "上の者", "社長"]


# ─── Supabase (lazy init to avoid module-level crash) ──────────────────────

def _get_supabase():
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )


async def get_concierge_config(called_number: str) -> Optional[dict]:
    try:
        sb = _get_supabase()
        pn = sb.from_("phone_numbers") \
            .select("id, tenant_id") \
            .eq("number", called_number) \
            .eq("is_active", True) \
            .limit(1).execute()
        if not pn.data:
            return None
        phone_number_id = pn.data[0]["id"]
        tenant_id = pn.data[0]["tenant_id"]

        cc = sb.from_("concierge_configs") \
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
        sb = _get_supabase()
        sb.from_("ai_conversations").insert({
            "tenant_id": config["tenant_id"],
            "transcript_json": transcript,
            "outcome": outcome,
            "gemini_model_used": GEMINI_MODEL,
            "livekit_room_id": livekit_room_id,
            "duration_seconds": duration_seconds,
        }).execute()
    except Exception as e:
        logger.error(f"save_ai_conversation error: {e}")


# ─── Conversation bridge ────────────────────────────────────────────────────

async def run_conversation(
    ctx: JobContext,
    config: Optional[dict],
    audio_source: rtc.AudioSource,
    audio_track: rtc.RemoteAudioTrack,
) -> tuple[list[dict], str]:

    company = (config or {}).get("company_name") or "弊社"
    greeting_tmpl = (config or {}).get("greeting_template") or DEFAULT_GREETING
    greeting = greeting_tmpl.replace("{company}", company)
    system_prompt = (config or {}).get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    escalation_kw = (config or {}).get("escalation_keywords") or DEFAULT_ESCALATION_KEYWORDS
    voice = (config or {}).get("voice") or "Zephyr"

    full_instructions = (
        f"{system_prompt}\n"
        f"会社名: {company}\n\n"
        f"【重要】会話が始まったら、ユーザーの発言を待たずに即座に次のセリフで挨拶してください:\n"
        f"「{greeting}」"
    )

    transcript: list[dict] = []
    escalated = False

    # Queue: Gemini → LiveKit (PCM bytes at 24kHz)
    audio_in_queue: asyncio.Queue[bytes] = asyncio.Queue()
    # Queue: LiveKit → Gemini (dict with data + mime_type)
    out_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=5)

    # LiveKit audio input stream (resampled to 16kHz by SDK)
    audio_stream = rtc.AudioStream(
        audio_track,
        sample_rate=SEND_SAMPLE_RATE,
        num_channels=CHANNELS,
    )

    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options={"api_version": "v1beta"},
    )

    live_config = genai_types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=genai_types.Content(
            parts=[genai_types.Part(text=full_instructions)],
        ),
        speech_config=genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=voice,
                )
            )
        ),
        realtime_input_config=genai_types.RealtimeInputConfig(
            # 話し始めたら即座にAI発話を中断する
            activity_handling=genai_types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            automatic_activity_detection=genai_types.AutomaticActivityDetection(
                start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=200,   # 発話開始を確定するまでの時間(ms) — 小さいほど敏感
                silence_duration_ms=400, # 発話終了を確定するまでの無音時間(ms)
            ),
        ),
    )

    room_disconnected = asyncio.Event()
    ctx.room.on("disconnected", lambda *_: room_disconnected.set())

    # ── タスク関数 ──────────────────────────────────────────────────────────

    async def listen_audio():
        """LiveKit音声フレームを out_queue に積む (16kHz PCM)"""
        logger.info("listen_audio: started")
        frames = 0
        try:
            async for frame_event in audio_stream:
                data = bytes(frame_event.frame.data)
                frames += 1
                if frames == 1:
                    logger.info("listen_audio: first frame received from LiveKit")
                try:
                    out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
                except asyncio.QueueFull:
                    # リアルタイム性維持: 最古フレームを捨てて新しいフレームを入れる
                    out_queue.get_nowait()
                    out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("listen_audio error", exc_info=True)
            raise
        finally:
            logger.info(f"listen_audio: ended (frames={frames})")

    async def send_realtime(session):
        """out_queue から Gemini に音声を送る"""
        logger.info("send_realtime: started")
        sent = 0
        try:
            while True:
                msg = await out_queue.get()
                await session.send_realtime_input(audio=msg)
                sent += 1
                if sent == 1:
                    logger.info("send_realtime: first audio chunk sent to Gemini")
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("send_realtime error", exc_info=True)
            raise
        finally:
            logger.info(f"send_realtime: ended (sent={sent})")

    async def receive_audio(session):
        """Gemini から音声を受け取り audio_in_queue に積む (ターン単位ループ)"""
        nonlocal escalated
        logger.info("receive_audio: started")
        total_chunks = 0
        try:
            while True:
                turn = session.receive()
                async for response in turn:
                    # response.data: 24kHz PCM バイト列 (inline_data shortcut)
                    if response.data:
                        total_chunks += 1
                        if total_chunks == 1:
                            logger.info(f"receive_audio: first audio chunk from Gemini ({len(response.data)} bytes)")
                        audio_in_queue.put_nowait(response.data)

                    # response.text: テキスト転写
                    if response.text:
                        logger.info(f"receive_audio: agent text: {response.text!r}")
                        transcript.append({
                            "role": "agent",
                            "text": response.text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        for kw in escalation_kw:
                            if kw in response.text:
                                escalated = True

                # ターン完了 — 割り込み時に残留音声をフラッシュ
                logger.info(f"receive_audio: turn complete (total_chunks={total_chunks})")
                while not audio_in_queue.empty():
                    audio_in_queue.get_nowait()

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("receive_audio error", exc_info=True)
            raise
        finally:
            logger.info(f"receive_audio: ended (total_chunks={total_chunks})")

    async def play_audio():
        """audio_in_queue の24kHz PCM を LiveKit AudioSource に流す"""
        logger.info("play_audio: started")
        played = 0
        try:
            while True:
                pcm = await audio_in_queue.get()
                played += 1
                frame = rtc.AudioFrame(
                    data=pcm,
                    sample_rate=RECV_SAMPLE_RATE,  # 24kHz (Gemini output)
                    num_channels=CHANNELS,
                    samples_per_channel=len(pcm) // 2,
                )
                await audio_source.capture_frame(frame)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("play_audio error", exc_info=True)
            raise
        finally:
            logger.info(f"play_audio: ended (played={played})")

    async def trigger_greeting(session):
        """receive_audio が起動してから send_client_content で挨拶を指示する"""
        await asyncio.sleep(0.5)  # receive_audio ループが確実に起動するのを待つ
        try:
            await session.send_client_content(
                turns=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="こんにちは")],
                ),
                turn_complete=True,
            )
            logger.info("trigger_greeting: send_client_content sent")
        except Exception:
            logger.error("trigger_greeting: send_client_content failed", exc_info=True)

    # ── セッション確立 + タスク実行 ─────────────────────────────────────────

    try:
        async with client.aio.live.connect(
            model=GEMINI_MODEL, config=live_config
        ) as session:
            logger.info("Gemini Live session established")

            # greeting は完了しても会話を続けるため wait に含めない
            asyncio.create_task(trigger_greeting(session), name="greeting")

            wait_tasks = [
                asyncio.create_task(receive_audio(session), name="recv"),
                asyncio.create_task(play_audio(),           name="play"),
                asyncio.create_task(listen_audio(),         name="listen"),
                asyncio.create_task(send_realtime(session), name="send"),
            ]
            disc_task = asyncio.create_task(room_disconnected.wait(), name="disconnect")

            done, pending = await asyncio.wait(
                wait_tasks + [disc_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in done:
                if not t.cancelled() and t.exception():
                    logger.error(
                        f"Task '{t.get_name()}' failed: "
                        f"{type(t.exception()).__name__}: {t.exception()}"
                    )
                else:
                    logger.info(f"Task '{t.get_name()}' completed first")

            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    except Exception:
        logger.error("run_conversation: session error", exc_info=True)

    outcome = "escalated" if escalated else ("resolved" if transcript else "abandoned")
    logger.info(f"run_conversation: done outcome={outcome}")
    return transcript, outcome


# ─── Entrypoint ────────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    logger.info(f"Job started: room={ctx.room.name}")
    await ctx.connect()

    start_time = datetime.now(timezone.utc)

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

    # AudioSource は Gemini の出力サンプルレート (24kHz) で作成
    audio_source = rtc.AudioSource(RECV_SAMPLE_RATE, CHANNELS)
    local_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
    await ctx.room.local_participant.publish_track(local_track)

    # SIP 参加者の音声トラックを待機 (最大15秒)
    audio_track: Optional[rtc.RemoteAudioTrack] = None
    for _ in range(30):
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


# ─── Worker ────────────────────────────────────────────────────────────────

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
