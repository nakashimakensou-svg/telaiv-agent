"""
Telaiv 日次通話レポート — Railway Cron から毎朝8時(JST)に実行
python report.py
"""

import os
print(f"CWD: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import aiohttp
from dotenv import load_dotenv

load_dotenv()

JST = timezone(timedelta(hours=9))


# ── Supabase REST API ─────────────────────────────────────────────────────────

def _sb_headers() -> dict:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}"}


async def _sb_get(session: aiohttp.ClientSession, path: str, params) -> list:
    url = os.environ["SUPABASE_URL"].rstrip("/") + f"/rest/v1/{path}"
    async with session.get(url, params=params, headers=_sb_headers()) as resp:
        return await resp.json()


async def _sb_post(session: aiohttp.ClientSession, path: str, data: dict) -> None:
    url = os.environ["SUPABASE_URL"].rstrip("/") + f"/rest/v1/{path}"
    async with session.post(url, json=data, headers={
        **_sb_headers(), "Content-Type": "application/json"
    }) as resp:
        if resp.status not in (200, 201):
            body = await resp.text()
            print(f"_sb_post {path}: status={resp.status} body={body[:200]}")


# ── 天気取得 ───────────────────────────────────────────────────────────────────

async def fetch_weather(session: aiohttp.ClientSession) -> str:
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "不明"
    try:
        async with session.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": "Fukuoka,JP", "appid": api_key, "lang": "ja", "units": "metric"},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            data = await resp.json()
            desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            return f"{desc} {temp:.0f}℃"
    except Exception as e:
        print(f"fetch_weather error: {e}")
        return "不明"


# ── Slack 送信 ────────────────────────────────────────────────────────────────

async def send_slack(message: str) -> bool:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("SLACK_WEBHOOK_URL not set")
        return False
    async with aiohttp.ClientSession() as session:
        async with session.post(
            webhook,
            json={"text": message},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            status = resp.status
            body = await resp.text()
            print(f"send_slack: status={status} body={body!r}")
            return status == 200


# ── レポート生成 ───────────────────────────────────────────────────────────────

def _parse_jst_hour(called_at: str) -> int:
    """ISO 8601 文字列 → JST の時刻（0-23）"""
    try:
        dt = datetime.fromisoformat(called_at.replace("Z", "+00:00"))
        return dt.astimezone(JST).hour
    except Exception:
        return 0


def _fmt_duration(seconds: int | None) -> str:
    s = seconds or 0
    return f"{s // 60}分{s % 60:02d}秒"


def _sentiment_emoji(s: str | None) -> str:
    return {"positive": "🟢", "neutral": "🟡", "negative": "🟠", "angry": "🔴"}.get(s or "", "⚪")


async def build_and_send_report(target_date: datetime) -> None:
    date_str = target_date.strftime("%Y-%m-%d")
    date_label = target_date.strftime("%Y年%m月%d日（%a）")

    print(f"build_and_send_report: target={date_str}")

    async with aiohttp.ClientSession() as session:
        weather = await fetch_weather(session)

        # 当日の通話履歴（JST 日付 = UTC で前後にまたがる可能性があるので余裕を持って取得）
        memories: list[dict] = await _sb_get(session, "call_memories", [
            ("called_at", f"gte.{date_str}T00:00:00+09:00"),
            ("called_at", f"lte.{date_str}T23:59:59+09:00"),
            ("order", "called_at.asc"),
        ])

        # callers テーブル（リピーター情報・通話回数）
        callers: list[dict] = await _sb_get(session, "callers", [
            ("order", "call_count.desc"),
        ])

    callers_by_phone = {c["phone_number"]: c for c in callers}

    def caller_name(phone: str | None) -> str:
        if not phone:
            return "不明"
        c = callers_by_phone.get(phone)
        if not c:
            return "不明"
        return c.get("contact_name") or c.get("company_name") or "不明"

    total = len(memories)
    NL = "\n"

    # ── 通話ゼロ ─────────────────────────────────────────────────────────────
    if total == 0:
        msg = (
            f"📊 *{date_label} 通話日次レポート*\n"
            f"天気: {weather}\n\n"
            f"本日の着信はありませんでした。"
        )
        await send_slack(msg)
        await _record_sent(date_str)
        return

    # ── 集計 ─────────────────────────────────────────────────────────────────
    sentiments = [m.get("sentiment") for m in memories]
    positive_count  = sentiments.count("positive")
    neutral_count   = sentiments.count("neutral")
    negative_count  = sentiments.count("negative")
    angry_count     = sentiments.count("angry")

    avg_duration = sum(m.get("duration_seconds") or 0 for m in memories) / total

    # ピーク時間帯
    hour_dist: dict[int, int] = {}
    for m in memories:
        h = _parse_jst_hour(m.get("called_at") or "")
        hour_dist[h] = hour_dist.get(h, 0) + 1
    peak_hour_num = max(hour_dist, key=lambda h: hour_dist[h])
    peak_hour = f"{peak_hour_num:02d}時台"

    # リピーター（call_count >= 2）
    repeat_count = sum(
        1 for m in memories
        if callers_by_phone.get(m.get("phone_number", ""), {}).get("call_count", 0) >= 2
    )
    repeat_rate = repeat_count / total * 100

    # 未対応（sentiment が negative/angry かつ responded=False）
    unresponded = [
        m for m in memories
        if m.get("sentiment") in ("negative", "angry") and not m.get("responded")
    ]

    # ── 未対応セクション ─────────────────────────────────────────────────────
    if unresponded:
        lines = []
        for m in unresponded:
            phone = m.get("phone_number") or "番号不明"
            name = caller_name(phone)
            time_str = (m.get("called_at") or "")[:16].replace("T", " ")
            try:
                dt = datetime.fromisoformat(
                    (m.get("called_at") or "").replace("Z", "+00:00")
                )
                time_str = dt.astimezone(JST).strftime("%H:%M")
            except Exception:
                time_str = "不明"
            summary = (m.get("summary") or "")[:40]
            urgency = m.get("urgency") or "medium"
            lines.append(f"  • {time_str} {name}（{phone}）: {summary} [緊急度:{urgency}]")
        unresponded_section = (
            f"\n⚠️ *未対応案件 {len(unresponded)}件* ← 要折り返し\n"
            + NL.join(lines)
            + "\n"
        )
    else:
        unresponded_section = "\n✅ 未対応案件なし\n"

    # ── 通話一覧（直近10件）────────────────────────────────────────────────
    call_list_lines = []
    for m in memories[-10:]:
        phone = m.get("phone_number") or "不明"
        name = caller_name(phone)
        try:
            dt = datetime.fromisoformat(
                (m.get("called_at") or "").replace("Z", "+00:00")
            )
            t = dt.astimezone(JST).strftime("%H:%M")
        except Exception:
            t = "--:--"
        emoji = _sentiment_emoji(m.get("sentiment"))
        dur = _fmt_duration(m.get("duration_seconds"))
        call_list_lines.append(f"  {emoji} {t} {name} {dur}")

    # ── リピーター上位5 ───────────────────────────────────────────────────────
    top_callers = [c for c in callers if (c.get("call_count") or 0) >= 2][:5]
    if top_callers:
        repeat_lines = [
            f"  • {c.get('company_name') or c.get('contact_name') or c['phone_number']}: "
            f"{c['call_count']}回"
            for c in top_callers
        ]
    else:
        repeat_lines = ["  なし"]

    # ── メッセージ組み立て ────────────────────────────────────────────────────
    message = (
        f"📊 *{date_label} 通話日次レポート*\n"
        f"天気: {weather}\n\n"
        f"━━━━━━━━━━━━━━\n"
        f"📞 *本日の通話サマリー*\n"
        f"- 総着信数: *{total}件*\n"
        f"- 平均通話時間: {_fmt_duration(int(avg_duration))}\n"
        f"- ピーク時間帯: {peak_hour}\n"
        f"- リピーター率: {repeat_rate:.0f}%\n\n"
        f"😊 *感情分布*\n"
        f"🟢 positive: {positive_count}件\n"
        f"🟡 neutral: {neutral_count}件\n"
        f"🟠 negative: {negative_count}件\n"
        f"🔴 angry: {angry_count}件\n"
        f"{unresponded_section}"
        f"━━━━━━━━━━━━━━\n"
        f"📋 *通話一覧（直近10件）*\n"
        f"{NL.join(call_list_lines)}\n\n"
        f"━━━━━━━━━━━━━━\n"
        f"🔁 *累計リピーター上位*\n"
        f"{NL.join(repeat_lines)}"
    )

    await send_slack(message)
    await _record_sent(date_str)


async def _record_sent(date_str: str) -> None:
    """daily_reports テーブルに送信記録を残す"""
    try:
        async with aiohttp.ClientSession() as session:
            await _sb_post(session, "daily_reports", {
                "report_date": date_str,
                "slack_sent_at": datetime.now(timezone.utc).isoformat(),
            })
        print(f"_record_sent: recorded {date_str}")
    except Exception as e:
        print(f"_record_sent error: {e}")


# ── エントリポイント ───────────────────────────────────────────────────────────

async def main() -> None:
    now = datetime.now(JST)
    # Railway Cron は前日分（UTC 23:00 = JST 翌8:00）に実行されるため前日を対象にする
    target = now - timedelta(days=1)
    print(f"main: now={now.isoformat()} target={target.strftime('%Y-%m-%d')}")
    await build_and_send_report(target)


if __name__ == "__main__":
    import sys
    asyncio.run(main())
    sys.exit(0)

