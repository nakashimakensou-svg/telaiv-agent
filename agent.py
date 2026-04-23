"""
Telaiv LiveKit Agent
Telnyx SIP着信 → LiveKit Room → Gemini Live リアルタイム音声AI
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
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import google, silero

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telaiv-agent")

# ─── Supabase クライアント ────────────────────────────────────────────────────
supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)

# ─── デフォルト設定 ──────────────────────────────────────────────────────────
DEFAULT_GREETING = "はい、お電話ありがとうございます。"
DEFAULT_SYSTEM_PROMPT = """\
あなたは企業のAI受付担当です。
電話に出て、お客様の用件を丁寧に伺い、適切に対応してください。

対応ルール:
- 常に丁寧な敬語を使う
- 用件を正確に聞き取りメモを取る
- 担当者への転送が必要な場合は「担当者におつなぎします」と伝える
- 営業時間外の場合は折り返し連絡の旨を伝える
- 30秒を超える沈黙があれば「もしもし、聞こえていますか？」と確認する
"""
DEFAULT_ESCALATION_KEYWORDS = ["クレーム", "担当者", "責任者", "上の者", "社長"]


# ─── Supabase ヘルパー ────────────────────────────────────────────────────────

async def get_concierge_config(called_number: str) -> Optional[dict]:
    """着信番号からconcierge_configを取得する"""
    try:
        pn = supabase.from_("phone_numbers") \
            .select("id, tenant_id") \
            .eq("number", called_number) \
            .eq("is_active", True) \
            .limit(1) \
            .execute()

        if not pn.data:
            logger.warning(f"Phone number not found: {called_number}")
            return None

        phone_number_id = pn.data[0]["id"]
        tenant_id = pn.data[0]["tenant_id"]

        cc = supabase.from_("concierge_configs") \
            .select("*") \
            .eq("phone_number_id", phone_number_id) \
            .eq("is_enabled", True) \
            .limit(1) \
            .execute()

        if not cc.data:
            logger.warning(f"No enabled concierge config for {called_number}")
            # デフォルト設定でテナント情報だけ付与して返す
            return {"tenant_id": tenant_id, "phone_number_id": phone_number_id}

        return cc.data[0]

    except Exception as e:
        logger.error(f"get_concierge_config error: {e}")
        return None


async def save_ai_conversation(
    config: Optional[dict],
    call_control_id: Optional[str],
    transcript: list[dict],
    outcome: str,
    duration_seconds: int,
    gemini_model: str,
    livekit_room_id: str,
) -> None:
    """会話ログをai_conversationsテーブルに保存する"""
    if not config or not config.get("tenant_id"):
        return
    try:
        # call_logsのIDをtelnyx_call_idで検索
        call_log_id = None
        if call_control_id:
            cl = supabase.from_("call_logs") \
                .select("id") \
                .eq("telnyx_call_id", call_control_id) \
                .limit(1) \
                .execute()
            if cl.data:
                call_log_id = cl.data[0]["id"]

        supabase.from_("ai_conversations").insert({
            "tenant_id": config["tenant_id"],
            "call_log_id": call_log_id,
            "transcript_json": transcript,
            "outcome": outcome,
            "gemini_model_used": gemini_model,
            "livekit_room_id": livekit_room_id,
            "duration_seconds": duration_seconds,
        }).execute()

        logger.info(f"ai_conversation saved for tenant={config['tenant_id']}")
    except Exception as e:
        logger.error(f"save_ai_conversation error: {e}")


# ─── エージェント ─────────────────────────────────────────────────────────────

class TelaivAssistant(Agent):
    """Gemini Live を使った AI 受付エージェント"""

    def __init__(self, config: Optional[dict]) -> None:
        self.config = config or {}
        self._transcript: list[dict] = []
        self._escalated = False
        self._start_time = datetime.now(timezone.utc)

        company = self.config.get("company_name") or "弊社"
        greeting_tmpl = self.config.get("greeting_template") or DEFAULT_GREETING
        self._greeting = greeting_tmpl.replace("{company}", company)

        escalation_kw = self.config.get("escalation_keywords") or DEFAULT_ESCALATION_KEYWORDS
        self._escalation_keywords: list[str] = escalation_kw

        system_prompt = self.config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        full_prompt = f"""\
{system_prompt}

会社名: {company}
エスカレーションキーワード（これらが出たら担当者転送）: {', '.join(self._escalation_keywords)}
"""
        super().__init__(instructions=full_prompt)

    async def on_enter(self) -> None:
        """ルームに入ったらグリーティングを再生"""
        logger.info(f"Agent entered room, greeting: {self._greeting}")
        await self.session.say(self._greeting, allow_interruptions=True)

    async def on_user_turn_completed(
        self, turn_ctx, new_message
    ) -> None:
        """ユーザー発話後にエスカレーションキーワードをチェック"""
        text = new_message.text_content or ""
        self._transcript.append({
            "role": "user",
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        for kw in self._escalation_keywords:
            if kw in text:
                logger.info(f"Escalation keyword detected: {kw}")
                self._escalated = True
                await self.session.say(
                    "少々お待ちください。担当者におつなぎします。",
                    allow_interruptions=False,
                )
                await self.session.room.disconnect()
                return

    def add_agent_message(self, text: str) -> None:
        self._transcript.append({
            "role": "agent",
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_outcome(self) -> str:
        if self._escalated:
            return "escalated"
        if self._transcript:
            return "resolved"
        return "abandoned"

    def get_duration(self) -> int:
        delta = datetime.now(timezone.utc) - self._start_time
        return int(delta.total_seconds())


# ─── エントリーポイント ───────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    logger.info(f"Job started: room={ctx.room.name}")

    await ctx.connect()

    # ── 着信番号の取得 ────────────────────────────────────────────────────────
    # LiveKit SIP dispatch rule でルームメタデータに called_number を埋め込む想定
    # フォールバック: SIP参加者のattributesを確認
    called_number: Optional[str] = None
    call_control_id: Optional[str] = None

    try:
        room_meta = json.loads(ctx.room.metadata or "{}")
        called_number = room_meta.get("called_number") or room_meta.get("to")
        call_control_id = room_meta.get("call_control_id")
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
            call_control_id = attrs.get("call_control_id") or call_control_id
            if called_number:
                logger.info(f"Called number from participant attrs: {called_number}")
                break

    logger.info(f"called_number={called_number}, call_control_id={call_control_id}")

    # ── Concierge 設定取得 ────────────────────────────────────────────────────
    config: Optional[dict] = None
    if called_number:
        config = await get_concierge_config(called_number)
        logger.info(f"concierge_config loaded: {bool(config)}")

    # ── Gemini Live モデル選択 ────────────────────────────────────────────────
    voice_map = {
        "Puck": "Puck", "Charon": "Charon",
        "Kore": "Kore", "Aoede": "Aoede", "Fenrir": "Fenrir",
    }
    voice = voice_map.get((config or {}).get("voice", "Puck"), "Puck")
    model_id = (config or {}).get("model_version") or "gemini-2.0-flash-exp"
    fallback_id = (config or {}).get("fallback_model") or "gemini-2.0-flash-exp"

    logger.info(f"Using Gemini model={model_id}, voice={voice}")

    # ── エージェント起動 ──────────────────────────────────────────────────────
    assistant = TelaivAssistant(config=config)

    try:
        realtime_model = google.beta.realtime.RealtimeModel(
            model=model_id,
            voice=voice,
            api_key=os.environ["GOOGLE_API_KEY"],
            instructions=assistant.instructions,
        )
    except Exception as e:
        logger.warning(f"Primary model {model_id} failed ({e}), trying fallback {fallback_id}")
        realtime_model = google.beta.realtime.RealtimeModel(
            model=fallback_id,
            voice=voice,
            api_key=os.environ["GOOGLE_API_KEY"],
            instructions=assistant.instructions,
        )

    session = AgentSession(
        llm=realtime_model,
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=True,
        ),
    )

    # ── 通話終了まで待機してから後処理 ───────────────────────────────────────
    try:
        await ctx.wait_for_disconnect()
    finally:
        logger.info("Call ended, saving conversation log")
        await save_ai_conversation(
            config=config,
            call_control_id=call_control_id,
            transcript=assistant._transcript,
            outcome=assistant.get_outcome(),
            duration_seconds=assistant.get_duration(),
            gemini_model=model_id,
            livekit_room_id=ctx.room.name,
        )
        logger.info("Job completed")


# ─── Worker 起動 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="telaiv-agent",
        )
    )
