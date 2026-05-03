"""DB-driven business hours utilities for Telaiv agent.

Priority order for business hours determination:
  1. schedule JSONB from ivr_configs (DB-first)
  2. BUSINESS_HOURS env var (fallback)
  3. 平日 08:00-17:00 (hardcoded last resort)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("telaiv.business_hours")

JST = timezone(timedelta(hours=9))

_WEEKDAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]


def is_within_business_hours(schedule: Optional[dict]) -> bool:
    """Check if current JST time is within business hours.

    Args:
        schedule: the `schedule` JSONB from ivr_configs, or None.
            Expected shape: {
              "business_hours": {
                "monday": {"enabled": True, "start": "09:00", "end": "18:00"},
                ...
              },
              "after_hours_action": "voicemail",
              "transfer_number": "+819012345678"
            }

    Falls back to BUSINESS_HOURS env var then hardcoded default.
    """
    if schedule:
        bh = schedule.get("business_hours", {})
        if bh:
            now = datetime.now(JST)
            day_name = _WEEKDAY_NAMES[now.weekday()]
            day_cfg = bh.get(day_name, {})
            if not day_cfg.get("enabled", False):
                return False
            try:
                sh, sm = map(int, day_cfg.get("start", "09:00").split(":"))
                eh, em = map(int, day_cfg.get("end", "18:00").split(":"))
            except (ValueError, AttributeError):
                logger.warning(
                    f"is_within_business_hours: invalid time format in schedule "
                    f"for {day_name}, falling back to env"
                )
                return _env_is_business_hours()
            current = now.hour * 60 + now.minute
            return (sh * 60 + sm) <= current < (eh * 60 + em)

    return _env_is_business_hours()


def get_after_hours_action(schedule: Optional[dict]) -> str:
    """Return the after_hours_action from schedule JSONB.

    Returns: 'voicemail' | 'transfer' | 'announcement'
    Default: 'voicemail'
    """
    if not schedule:
        return "voicemail"
    return schedule.get("after_hours_action") or "voicemail"


def get_transfer_number(schedule: Optional[dict]) -> Optional[str]:
    """Return the transfer phone number from schedule JSONB, or None."""
    if not schedule:
        return None
    return schedule.get("transfer_number") or None


def get_fallback_action(ivr_config: Optional[dict]) -> str:
    """Return the fallback_action from an ivr_config row dict.

    Returns: 'voicemail' | 'transfer' | 'announcement'
    Default: 'voicemail'
    """
    if not ivr_config:
        return "voicemail"
    return ivr_config.get("fallback_action") or "voicemail"


def _env_is_business_hours() -> bool:
    """Fallback: reads BUSINESS_HOURS env var (same logic as original agent_genai.py)."""
    raw = os.environ.get("BUSINESS_HOURS", "1-5,08:00-17:00")
    try:
        days_part, time_part = raw.split(",")
        d_start, d_end = map(int, days_part.split("-"))
        t_start_str, t_end_str = time_part.split("-")
        sh, sm = map(int, t_start_str.split(":"))
        eh, em = map(int, t_end_str.split(":"))
    except (ValueError, AttributeError):
        logger.warning(
            f"_env_is_business_hours: Invalid BUSINESS_HOURS env: {raw!r}, "
            "using default 1-5,08:00-17:00"
        )
        d_start, d_end, sh, sm, eh, em = 1, 5, 8, 0, 17, 0

    now = datetime.now(JST)
    weekday = now.isoweekday()  # Mon=1 ... Sun=7
    if not (d_start <= weekday <= d_end):
        return False
    current = now.hour * 60 + now.minute
    return (sh * 60 + sm) <= current < (eh * 60 + em)
