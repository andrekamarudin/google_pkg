import re
from datetime import datetime

import pytz


def condense_text(text):
    if not text:
        return text
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\n+\s*", " | ", text)
    return text


def get_time_passed(timestamp_dt) -> str:
    now = datetime.now(pytz.timezone("Asia/Singapore"))
    dt_since = now - timestamp_dt
    units = [
        ("yr", 31536000),
        ("mth", 2592000),
        ("day", 86400),
        ("hr", 3600),
        ("min", 60),
        ("sec", 1),
    ]
    seconds = dt_since.total_seconds()
    text = ""
    count = 0

    for unit, value in units:
        if count < 2:
            unit_value, seconds = divmod(seconds, value)
            if unit_value > 0:
                text += f"{int(unit_value)} {unit}{'s' if unit_value > 1 else ''} "
                count += 1
    text += "ago"
    return text
