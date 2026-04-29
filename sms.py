from __future__ import annotations

import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

TELNYX_API_URL = "https://api.telnyx.com/v2/messages"


async def send_sms(to: str, message: str) -> bool:
    api_key = os.environ.get("TELNYX_API_KEY")
    from_number = os.environ.get("TELNYX_SMS_FROM")

    if not api_key or not from_number:
        logger.warning("send_sms: TELNYX_API_KEY or TELNYX_SMS_FROM not set")
        return False

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TELNYX_API_URL,
                json={"from": from_number, "to": to, "text": message},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                body = await resp.text()
                if resp.status in (200, 201):
                    logger.info(f"send_sms: sent to={to} status={resp.status}")
                    return True
                else:
                    logger.error(
                        f"send_sms: FAILED to={to} status={resp.status} body={body[:100]}"
                    )
                    return False
    except Exception as e:
        logger.error(f"send_sms: exception {e}")
        return False


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
