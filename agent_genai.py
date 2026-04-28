"""
Telaiv LiveKit Agent — google-genai SDK direct implementation
Based on official Google Gemini cookbook Live API examples.
https://github.com/google-gemini/cookbook/tree/main/quickstarts
"""

from __future__ import annotations

import array
import asyncio
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telaiv-agent-genai")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview")

# Gemini Live I/O sample rates (per official cookbook)
SEND_SAMPLE_RATE = 16000   # LiveKit → Gemini: 16kHz PCM
RECV_SAMPLE_RATE = 24000   # Gemini → LiveKit: 24kHz PCM
CHANNELS = 1

JST = timezone(timedelta(hours=9))

# 営業電話フィルタリングキーワード
FILTER_KEYWORDS = [
    "広告", "営業", "アンケート", "保険", "投資",
    "不動産投資", "太陽光", "ご提案", "ご案内", "キャンペーン",
]
# 時間外緊急対応キーワード
EMERGENCY_KEYWORDS = ["雨漏り", "水漏れ", "緊急", "事故"]

DEFAULT_GREETING = "はい、お電話ありがとうございます。"
DEFAULT_ESCALATION_KEYWORDS = ["クレーム", "担当者", "責任者", "上の者", "社長"]

# ─── 汎用対応ルール（全テナント共通・上書き不可）─────────────────────────────
# {business_hours_note} は run_conversation で動的に差し込む
DEFAULT_SYSTEM_PROMPT = """\
あなたは企業のAI電話受付担当です。人間のオペレーターのように自然な会話をしてください。

【基本姿勢】
- 常に丁寧な敬語を使い、落ち着いたトーンで話す
- 相手のペースに合わせ、決して急かさない
- 用件を正確に聞き取り、繰り返して確認する

【自然な会話の振る舞い】
- 「そうですか」「なるほど」「かしこまりました」など自然な相槌を入れる
- 相手が話し終わるまで遮らず、しっかり聞く
- 会話が途切れたら「いかがでしょうか？」「他にご用件はございますか？」と自然につなぐ

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【優先ルール①：営業・勧誘電話のフィルタリング】

以下のいずれかに当てはまると判断した場合は、
必ず下記のお断りセリフをそのまま読み上げ、即座に会話を終了してください。

▶ 判断基準
・「広告」「営業」「アンケート」「保険」「投資」「不動産投資」「太陽光」
  「ご提案」「ご案内」「キャンペーン」「モニター」が用件に含まれる
・「〜のサービスをご案内したい」「〜についてお話ししたい」など勧誘的な話し方

▶ お断りセリフ（必ずこのまま読む・一字一句変えない）
「誠に恐れ入りますが、現在そのようなご案内はお断りしております。
お電話ありがとうございました。失礼いたします。」

⚠ このルールに例外はありません。「少し聞いてみてから判断する」はしないこと。
⚠ 相手が「でも」「ちょっとだけ」と食い下がっても同じセリフで終了すること。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【優先ルール②：営業時間外の対応】

{business_hours_note}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【状況対応】
- 咳き込み・体調不良: 「大丈夫でしょうか？お気をつけください」と気遣う
- 無言・返答なし: 数秒待ってから「もしもし、聞こえていますか？」と確認
- 怒っている: 落ち着いたトーンで「大変申し訳ございません」と誠実に対応
- 高齢者・ゆっくり話す方: さらにゆっくり丁寧なトーンで対応

【禁止事項】
- 同じ質問を繰り返さない
- でたらめを言わない。不明な点は「担当者に確認いたします」と伝える
- 優先ルール①の営業電話に対して何度も確認したり対話を続けない
"""

# ─── 中島建装専用プロファイル（DB設定がない場合のデフォルト）─────────────────
DEFAULT_COMPANY_PROFILE = """\
【会社情報】
会社名: 中島建装
業務内容: 塗装工事・防水工事・外壁工事
対応エリア: 佐賀県・福岡県
担当者: よしき（社長）

【対応指針】
- 見積もり依頼: 「担当者より折り返しご連絡いたします。ご連絡先をお教えください。」
  → 名前・電話番号・ご希望日時を丁寧に聞き取る
- 施工・日程確認: 「担当者より確認してご連絡いたします」
- 緊急の雨漏り・水漏れ: 「緊急の対応が必要とのこと、かしこまりました。
  担当者に至急連絡いたします。少々お待ちください。」
- 営業電話: 優先ルール①に従い丁重にお断り（例外なし）
"""


# ─── 営業時間ユーティリティ ────────────────────────────────────────────────────

def is_business_hours() -> bool:
    """現在時刻が営業時間内かどうか（JST）。
    BUSINESS_HOURS 環境変数フォーマット: '1-5,08:00-17:00'（月=1, 日=7）
    デフォルト: 平日（月〜金）08:00〜17:00
    """
    raw = os.environ.get("BUSINESS_HOURS", "1-5,08:00-17:00")
    try:
        days_part, time_part = raw.split(",")
        d_start, d_end = map(int, days_part.split("-"))
        t_start_str, t_end_str = time_part.split("-")
        sh, sm = map(int, t_start_str.split(":"))
        eh, em = map(int, t_end_str.split(":"))
    except (ValueError, AttributeError):
        logger.warning(f"Invalid BUSINESS_HOURS: {raw!r}, using default 1-5,08:00-17:00")
        d_start, d_end, sh, sm, eh, em = 1, 5, 8, 0, 17, 0

    now = datetime.now(JST)
    weekday = now.isoweekday()  # Mon=1 ... Sun=7
    if not (d_start <= weekday <= d_end):
        return False
    current = now.hour * 60 + now.minute
    return (sh * 60 + sm) <= current < (eh * 60 + em)


def _build_business_hours_note() -> str:
    """現在の営業時間状況に応じた対応指示文を生成する"""
    raw = os.environ.get("BUSINESS_HOURS", "1-5,08:00-17:00")
    try:
        _, time_part = raw.split(",")
        t_start, t_end = time_part.split("-")
    except (ValueError, AttributeError):
        t_start, t_end = "08:00", "17:00"

    if is_business_hours():
        return f"現在は営業時間内（平日 {t_start}〜{t_end}）です。通常どおり対応してください。"

    return (
        f"現在は営業時間外です（営業時間: 平日 {t_start}〜{t_end}）。\n\n"
        "▶ 通常のお問い合わせ（見積もり・日程確認など）\n"
        "「ありがとうございます。現在は営業時間外のため、翌営業日に担当者よりご連絡いたします。\n"
        "ご用件とお名前・ご連絡先をお教えいただけますか？」と伝え、情報を丁寧に聞き取る。\n\n"
        "▶ 緊急対応（以下の条件を両方満たす場合のみ担当者連絡フラグを立てる）\n"
        "  条件1: 「雨漏り」「水漏れ」「緊急」「事故」のキーワードがある\n"
        "  条件2: 営業・広告目的でないと文脈から明らかに判断できる\n"
        "  → 「緊急の対応が必要とのこと、かしこまりました。担当者に至急連絡いたします。」と伝える\n\n"
        "⚠ 「緊急」と言っても用件が工事営業・保険提案であれば優先ルール①（フィルタリング）を適用すること。\n"
        "  文脈から営業目的の悪用と判断できる場合は通常のお断りに切り替えること。"
    )


# ─── 録音ユーティリティ ────────────────────────────────────────────────────────

PLAN_RETENTION_DAYS: dict[str, Optional[int]] = {
    "starter":    30,
    "business":   90,
    "pro":        180,
    "enterprise": None,  # 無制限
}


def _resample_pcm16(data: bytes, in_rate: int, out_rate: int) -> bytes:
    """int16 モノラル PCM を線形補間でリサンプリング (pure Python / array モジュール使用)"""
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
    """2 つの int16 モノラル PCM ストリームをミックス。短い方はゼロパディング。"""
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


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Raw int16 PCM → WAV バイト列 (stdlib wave モジュール使用)"""
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _upload_to_r2_sync(wav_data: bytes, key: str) -> str:
    """Cloudflare R2 (S3互換) にアップロードして公開 URL を返す (同期・スレッド実行用)"""
    import boto3
    from botocore.config import Config as BotoConfig

    account_id = os.environ["R2_ACCOUNT_ID"]
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )
    client.put_object(
        Bucket=os.environ["R2_BUCKET_NAME"],
        Key=key,
        Body=wav_data,
        ContentType="audio/wav",
    )
    public_url = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
    return f"{public_url}/{key}"


def _update_call_log_sync(
    telnyx_call_id: str,
    recording_url: str,
    expires_at: Optional[str],
) -> None:
    """call_logs.recording_url と recording_expires_at を更新 (同期・スレッド実行用)"""
    sb = _get_supabase()
    data: dict = {"recording_url": recording_url}
    if expires_at:
        data["recording_expires_at"] = expires_at
    sb.from_("call_logs") \
        .update(data) \
        .eq("telnyx_call_id", telnyx_call_id) \
        .execute()


async def save_recording(
    caller_chunks: list[bytes],
    ai_chunks: list[bytes],
    config: Optional[dict],
    room_name: str,
    telnyx_call_id: Optional[str],
) -> Optional[str]:
    """通話音声を混合 WAV にして R2 に保存し call_logs を更新する。
    - caller_chunks: listen_audio が収集した 16kHz PCM (着信側)
    - ai_chunks:     receive_audio が収集した 24kHz PCM (Gemini 側)
    """
    logger.info(
        f"save_recording: called "
        f"(caller={len(caller_chunks)} chunks, ai={len(ai_chunks)} chunks, "
        f"telnyx_call_id={telnyx_call_id})"
    )
    if not caller_chunks and not ai_chunks:
        logger.info("save_recording: no audio data, skipping")
        return None
    try:
        caller_pcm = b"".join(caller_chunks)          # 16kHz
        ai_pcm_raw = b"".join(ai_chunks)              # 24kHz
        ai_pcm = _resample_pcm16(ai_pcm_raw, RECV_SAMPLE_RATE, SEND_SAMPLE_RATE)
        mixed   = _mix_pcm16(caller_pcm, ai_pcm)
        wav_data = _pcm_to_wav(mixed, SEND_SAMPLE_RATE)

        ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        call_id = (telnyx_call_id or room_name).replace("/", "_")
        key     = f"recordings/{call_id}_{ts}.wav"

        logger.info(
            f"save_recording: uploading {key} "
            f"({len(wav_data) // 1024}KB | "
            f"caller={len(caller_pcm) // 1024}KB "
            f"ai={len(ai_pcm_raw) // 1024}KB)"
        )

        url = await asyncio.to_thread(_upload_to_r2_sync, wav_data, key)
        logger.info(f"save_recording: uploaded → {url}")

        if telnyx_call_id:
            plan = (config or {}).get("plan", "starter")
            retention = PLAN_RETENTION_DAYS.get(plan, 30)
            expires_at = (
                (datetime.now(timezone.utc) + timedelta(days=retention)).isoformat()
                if retention is not None else None
            )
            await asyncio.to_thread(
                _update_call_log_sync, telnyx_call_id, url, expires_at
            )
            logger.info(
                f"save_recording: call_logs updated "
                f"(plan={plan}, retention={retention}d, expires={expires_at})"
            )
        return url
    except Exception:
        logger.error("save_recording: failed", exc_info=True)
        return None


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

        tenant = sb.from_("tenants") \
            .select("plan") \
            .eq("id", tenant_id) \
            .limit(1).execute()
        plan = tenant.data[0]["plan"] if tenant.data else "starter"

        cc = sb.from_("concierge_configs") \
            .select("*") \
            .eq("phone_number_id", phone_number_id) \
            .eq("is_enabled", True) \
            .limit(1).execute()
        if not cc.data:
            return {"tenant_id": tenant_id, "phone_number_id": phone_number_id, "plan": plan}
        return {**cc.data[0], "plan": plan}
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


# ─── 感情分析ユーティリティ ────────────────────────────────────────────────────

async def _notify_slack(message: str) -> None:
    """Slack Webhook に非同期通知する（fire-and-forget 想定）"""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("_notify_slack: SLACK_WEBHOOK_URL not set")
        return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json={"text": message},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                status = resp.status
                body = await resp.text()
                if status == 200:
                    logger.info(f"_notify_slack: sent status=200 ({message[:80]!r})")
                else:
                    logger.error(
                        f"_notify_slack: FAILED status={status} body={body!r} "
                        f"url={webhook_url[:50]}..."
                    )
    except Exception:
        logger.error("_notify_slack: exception", exc_info=True)


def _update_call_log_ai_summary_sync(telnyx_call_id: str, ai_summary: dict) -> None:
    """call_logs.ai_summary を更新（同期・スレッド実行用）"""
    sb = _get_supabase()
    sb.from_("call_logs") \
        .update({"ai_summary": ai_summary}) \
        .eq("telnyx_call_id", telnyx_call_id) \
        .execute()


def _update_ai_conversation_sync(
    livekit_room_id: str,
    sentiment: str,
    ai_summary_text: str,
) -> None:
    """ai_conversations の sentiment と ai_summary を更新（同期・スレッド実行用）"""
    sb = _get_supabase()
    sb.from_("ai_conversations") \
        .update({"sentiment": sentiment, "ai_summary": ai_summary_text}) \
        .eq("livekit_room_id", livekit_room_id) \
        .execute()


async def post_call_analysis(
    transcript: list[dict],
    config: Optional[dict],
    telnyx_call_id: Optional[str],
    livekit_room_id: str,
    recording_url: Optional[str],
) -> Optional[dict]:
    """通話後に Claude API で会話全文を分析し、要対応の場合に Slack 通知する。
    - call_logs.ai_summary (JSONB) を更新
    - ai_conversations.sentiment + ai_summary を更新
    - sentiment が negative/angry、または is_complaint=True なら Slack 通知
    """
    if not transcript:
        logger.info("post_call_analysis: no transcript, skipping")
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("post_call_analysis: ANTHROPIC_API_KEY not set, skipping")
        return None

    lines = [
        f"{'発信者' if t['role'] == 'caller' else 'AI'}: {t['text']}"
        for t in transcript if t.get("text")
    ]
    if not lines:
        logger.info("post_call_analysis: transcript has no text, skipping")
        return None

    conversation_text = "\n".join(lines)
    prompt = (
        "以下は電話の会話記録です。発信者の感情・意図を分析してください。\n\n"
        f"【会話記録】\n{conversation_text}\n\n"
        "以下のJSON形式で回答してください（他の文字は不要）：\n"
        '{\n'
        '  "sentiment": "positive" | "neutral" | "negative" | "angry",\n'
        '  "is_complaint": true | false,\n'
        '  "urgency": "low" | "medium" | "high",\n'
        '  "summary": "会話の要約（1〜2文）",\n'
        '  "action_required": "対応が必要な場合の内容、不要なら空文字",\n'
        '  "keywords": ["重要キーワード"],\n'
        '  "call_type": "inquiry" | "complaint" | "appointment" | "other"\n'
        '}'
    )

    logger.info(
        f"post_call_analysis: calling Claude API "
        f"(lines={len(lines)}, model=claude-sonnet-4-6)"
    )
    raw_text = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                http_status = resp.status
                data = await resp.json()
                logger.info(f"post_call_analysis: Claude API responded status={http_status}")

        content = data.get("content", [])
        if not content:
            logger.error(f"post_call_analysis: empty Claude response data={data!r}")
            return None
        raw_text = content[0].get("text", "").strip()
        logger.info(f"post_call_analysis: raw Claude text={raw_text[:200]!r}")
        clean_text = (
            raw_text
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        result: dict = json.loads(clean_text)

        logger.info(
            f"post_call_analysis: sentiment={result.get('sentiment')} "
            f"is_complaint={result.get('is_complaint')} "
            f"urgency={result.get('urgency')}"
        )

        # call_logs.ai_summary 更新
        if telnyx_call_id:
            await asyncio.to_thread(
                _update_call_log_ai_summary_sync, telnyx_call_id, result
            )
        # ai_conversations 更新
        if livekit_room_id:
            await asyncio.to_thread(
                _update_ai_conversation_sync,
                livekit_room_id,
                result.get("sentiment", "neutral"),
                result.get("summary", ""),
            )

        # negative/angry または is_complaint=True の場合のみ Slack 通知
        sentiment = result.get("sentiment", "neutral")
        is_complaint = result.get("is_complaint", False)
        if sentiment in ("negative", "angry") or is_complaint:
            urgency = result.get("urgency", "medium")
            urgency_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(urgency, "⚠️")
            recording_line = f"録音: {recording_url}" if recording_url else "録音: なし"
            message = (
                f"{urgency_emoji} *要対応の電話がありました*\n"
                f"感情: {sentiment} | 緊急度: {urgency}\n"
                f"概要: {result.get('summary', '')}\n"
                f"対応内容: {result.get('action_required') or 'なし'}\n"
                f"{recording_line}"
            )
            await _notify_slack(message)

        return result
    except json.JSONDecodeError:
        logger.error(f"post_call_analysis: JSON parse error raw={raw_text!r}")
        return None
    except Exception as e:
        logger.error(
            f"post_call_analysis: FAILED type={type(e).__name__} msg={e}",
            exc_info=True,
        )
        return None


# ─── Conversation bridge ────────────────────────────────────────────────────

async def run_conversation(
    ctx: JobContext,
    config: Optional[dict],
    audio_source: rtc.AudioSource,
    audio_track: rtc.RemoteAudioTrack,
    caller_chunks: list[bytes],      # OUT: listen_audio が 16kHz PCM を蓄積
    ai_chunks: list[bytes],          # OUT: receive_audio が 24kHz PCM を蓄積
    telnyx_call_id: Optional[str],   # call_logs 更新用
    transcript: list[dict],          # OUT: entrypoint が作成した共有リスト（in-place 蓄積）
) -> str:                            # outcome のみ返す

    company = (config or {}).get("company_name") or "中島建装"
    greeting_tmpl = (config or {}).get("greeting_template") or DEFAULT_GREETING
    greeting = greeting_tmpl.replace("{company}", company)
    # company_profile: DB設定優先、なければ中島建装デフォルト
    company_profile = (config or {}).get("system_prompt") or DEFAULT_COMPANY_PROFILE
    escalation_kw = (config or {}).get("escalation_keywords") or DEFAULT_ESCALATION_KEYWORDS
    voice = (config or {}).get("voice") or "Zephyr"

    # 営業時間を呼び出しのタイミングで動的判定
    business_hours_note = _build_business_hours_note()
    base_rules = DEFAULT_SYSTEM_PROMPT.format(business_hours_note=business_hours_note)
    logger.info(f"business_hours: {'in' if is_business_hours() else 'out'}")

    full_instructions = (
        f"{base_rules}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"【担当会社プロファイル】\n"
        f"{company_profile}\n\n"
        f"会社名（挨拶・名乗りで使用）: {company}\n\n"
        f"【重要】会話が始まったら、ユーザーの発言を待たずに即座に次のセリフで挨拶してください:\n"
        f"「{greeting}」"
    )

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

    # input_audio_transcription: 発信者音声をテキスト変換（クレーム検知に必要）
    # SDKバージョンによっては未対応のため安全に設定
    _transcription_kwargs: dict = {}
    if hasattr(genai_types, "AudioTranscriptionConfig"):
        _transcription_kwargs["input_audio_transcription"] = genai_types.AudioTranscriptionConfig()

    live_config = genai_types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=genai_types.Content(
            parts=[genai_types.Part(text=full_instructions)],
        ),
        generation_config=genai_types.GenerationConfig(
            temperature=0.8,  # 安定した応答品質（デフォルト1.0より保守的）
            top_p=0.95,
        ),
        speech_config=genai_types.SpeechConfig(
            language_code="ja-JP",
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=voice,
                )
            ),
        ),
        realtime_input_config=genai_types.RealtimeInputConfig(
            # 話し始めたら即座にAI発話を中断する
            activity_handling=genai_types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            automatic_activity_detection=genai_types.AutomaticActivityDetection(
                start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=100,   # 200→100ms: 発話検知後より素早くAIを停止
                silence_duration_ms=300, # 400→300ms: 発話終了をより早く確定
            ),
        ),
        **_transcription_kwargs,
    )

    room_disconnected = asyncio.Event()
    ctx.room.on("disconnected", lambda *_: room_disconnected.set())

    # 割り込み時に play_audio へ即時フラッシュを指示するイベント
    _flush_evt: asyncio.Event = asyncio.Event()

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
                caller_chunks.append(data)  # 録音バッファに蓄積
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
                _flush_evt.clear()  # 新しいターン開始: フラッシュ解除
                turn = session.receive()
                is_interrupted = False

                async for response in turn:
                    sc = getattr(response, "server_content", None)

                    # server_content.interrupted: ユーザー発話による割り込み検知
                    if sc and getattr(sc, "interrupted", False):
                        is_interrupted = True
                        logger.info("receive_audio: interrupted by user speech")

                    # input_transcription: 発信者の音声テキスト変換（通話後Claude分析に使用）
                    if sc:
                        input_trans = getattr(sc, "input_transcription", None)
                        if input_trans:
                            caller_text = getattr(input_trans, "text", None)
                            if caller_text and caller_text.strip():
                                logger.info(f"receive_audio: caller text: {caller_text!r}")
                                transcript.append({
                                    "role": "caller",
                                    "text": caller_text,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })

                    # response.data: 24kHz PCM バイト列 (inline_data shortcut)
                    if response.data:
                        total_chunks += 1
                        if total_chunks == 1:
                            logger.info(f"receive_audio: first audio chunk from Gemini ({len(response.data)} bytes)")
                        ai_chunks.append(response.data)     # 録音バッファに蓄積
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

                # ターン完了 — play_audio に即時フラッシュを指示してからキューを空にする
                logger.info(
                    f"receive_audio: turn {'interrupted' if is_interrupted else 'complete'} "
                    f"(total_chunks={total_chunks})"
                )
                _flush_evt.set()
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
        """audio_in_queue の24kHz PCM を LiveKit AudioSource に流す。
        _flush_evt がセットされたら残留バッファを即座に破棄して再生を止める。"""
        logger.info("play_audio: started")
        played = 0
        try:
            while True:
                # 20ms タイムアウトで _flush_evt を定期チェックできるようにする
                try:
                    pcm = await asyncio.wait_for(audio_in_queue.get(), timeout=0.02)
                except asyncio.TimeoutError:
                    continue

                # フラッシュ指示が来ていたらこのフレームを含め残留音声を全破棄
                if _flush_evt.is_set():
                    while not audio_in_queue.empty():
                        audio_in_queue.get_nowait()
                    continue

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
        await asyncio.sleep(0.1)  # 0.5→0.1s: セッション確立直後に即挨拶
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
                    await asyncio.wait_for(asyncio.shield(t), timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
            logger.info("run_conversation: all tasks cancelled, exiting session")

    except asyncio.CancelledError:
        # LiveKit フレームワークによるキャンセル — re-raise せず finally で後処理を完走させる
        logger.info("run_conversation: CancelledError caught, running cleanup in finally")
    except Exception:
        logger.error("run_conversation: session error", exc_info=True)
    finally:
        logger.info("run_conversation: finally block executing")
        outcome = "escalated" if escalated else ("resolved" if transcript else "abandoned")
        logger.info(
            f"run_conversation: done outcome={outcome} "
            f"caller_chunks={len(caller_chunks)} ai_chunks={len(ai_chunks)} "
            f"transcript_len={len(transcript)}"
        )
        # save_recording / post_call_analysis は on_shutdown フックで実行

    return outcome


# ─── Entrypoint ────────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    logger.info(f"Job started: room={ctx.room.name}")
    await ctx.connect()

    start_time = datetime.now(timezone.utc)

    called_number: Optional[str] = None
    telnyx_call_id: Optional[str] = None
    try:
        room_meta = json.loads(ctx.room.metadata or "{}")
        called_number  = room_meta.get("called_number") or room_meta.get("to")
        telnyx_call_id = room_meta.get("telnyx_call_id") or room_meta.get("call_control_id")
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

    caller_chunks: list[bytes] = []
    ai_chunks: list[bytes] = []
    transcript: list[dict] = []      # run_conversation が in-place で蓄積する共有リスト

    # LiveKit がプロセス終了前に呼ぶシャットダウンフック
    # （finally ブロックが実行されない強制終了でも録音・分析を保証する）
    async def on_shutdown() -> None:
        logger.info(
            f"on_shutdown: triggered "
            f"(caller_chunks={len(caller_chunks)}, ai_chunks={len(ai_chunks)}, "
            f"transcript_len={len(transcript)})"
        )
        rec_url: Optional[str] = None
        try:
            rec_url = await save_recording(
                caller_chunks=caller_chunks,
                ai_chunks=ai_chunks,
                config=config,
                room_name=ctx.room.name,
                telnyx_call_id=telnyx_call_id,
            )
        except Exception:
            logger.error("on_shutdown: save_recording failed", exc_info=True)

        transcript_len = len(transcript)
        url_preview = rec_url[:60] if rec_url else "none"
        logger.info(
            f"on_shutdown: post_call_analysis starting "
            f"(transcript_len={transcript_len}, recording_url={url_preview})"
        )
        if transcript_len == 0:
            logger.warning(
                "on_shutdown: transcript empty — "
                "input_transcription 未取得か agent text が収集されていない可能性あり"
            )
        try:
            await post_call_analysis(
                transcript=transcript,
                config=config,
                telnyx_call_id=telnyx_call_id,
                livekit_room_id=ctx.room.name,
                recording_url=rec_url,
            )
        except Exception:
            logger.error("on_shutdown: post_call_analysis failed", exc_info=True)

    ctx.add_shutdown_callback(on_shutdown)

    outcome = await run_conversation(
        ctx, config, audio_source, audio_track,
        caller_chunks=caller_chunks,
        ai_chunks=ai_chunks,
        telnyx_call_id=telnyx_call_id,
        transcript=transcript,
    )

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
