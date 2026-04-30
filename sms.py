from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def _get_twilio_client():
    from twilio.rest import Client
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    token = os.environ["TWILIO_AUTH_TOKEN"]
    return Client(sid, token)


def _send_sms_sync(to: str, message: str) -> bool:
    from_number = os.environ.get("TWILIO_SMS_FROM", "+16504803451")
    try:
        client = _get_twilio_client()
        msg = client.messages.create(body=message, from_=from_number, to=to)
        logger.info(f"send_sms: sent to={to} sid={msg.sid} status={msg.status}")
        return True
    except Exception as e:
        logger.error(f"send_sms: FAILED to={to} error={e}")
        return False


async def send_sms(to: str, message: str) -> bool:
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        logger.warning("send_sms: TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN not set")
        return False
    return await asyncio.to_thread(_send_sms_sync, to, message)


def format_complaint_sms(
    phone_number: str,
    summary: str,
    urgency: str,
    company_name: str = "",
) -> str:
    urgency_label = {"high": "高", "medium": "中", "low": "低"}.get(urgency, "中")
    company_line = f"担当: {company_name}\n" if company_name else ""
    return (
        f"【Telaiv】クレーム電話\n"
        f"{company_line}"
        f"発信者: {phone_number}\n"
        f"緊急度: {urgency_label}\n"
        f"内容: {summary[:50]}\n"
        f"すぐ折り返してください"
    )


def format_daily_report_sms(
    date_label: str,
    total: int,
    angry_count: int,
    unresponded_count: int,
) -> str:
    return (
        f"【Telaiv】{date_label} 日次レポート\n"
        f"総着信: {total}件\n"
        f"クレーム: {angry_count}件\n"
        f"未対応: {unresponded_count}件"
    )
