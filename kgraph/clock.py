from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, field_validator


class IngestionClock(BaseModel, frozen=True):
    now: datetime

    @field_validator('now')
    @classmethod
    def now_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('IngestionClock value must be timezone-aware')
        return value
