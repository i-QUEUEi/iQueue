import re
from pathlib import Path

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def load_ph_holiday_month_days(calendar_path: Path):
    holiday_md = set()
    if not calendar_path.exists():
        return holiday_md

    text = calendar_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        match = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
        if not match:
            continue
        month_name = match.group(1).title()
        day = int(match.group(2))
        month = MONTH_MAP.get(month_name)
        if month:
            holiday_md.add((month, day))
    return holiday_md
