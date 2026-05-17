import re
from pathlib import Path

# Mapping of 3-letter month abbreviations to month numbers
# Used to parse holiday calendar entries like "Jan 1 : New Year's Day"
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
    """Parse a Philippine holiday calendar CSV and return a set of (month, day) tuples.

    The function reads a text file where holidays are listed as lines like:
        "Jan 1 : New Year's Day"
    It uses a regex to extract the month abbreviation and day number,
    then converts them to (month_number, day_number) tuples stored in a set.

    Using a set ensures O(1) lookup time — checking "is this date a holiday?"
    is instant regardless of how many holidays exist.

    Args:
        calendar_path: Path to the holiday calendar CSV file.

    Returns:
        A set of (month, day) tuples, e.g., {(1, 1), (6, 12), (12, 25), ...}.
        Returns an empty set if the file doesn't exist.
    """
    holiday_md = set()

    # If the calendar file is missing, return empty set (no holidays)
    if not calendar_path.exists():
        return holiday_md

    # Read the entire file as plain text
    text = calendar_path.read_text(encoding="utf-8", errors="ignore")

    # Process each line looking for holiday entries.
    # IMPORTANT: Each CSV line contains up to 3 holidays side-by-side
    # (one per column group), so we must use re.findall — not re.search —
    # to capture ALL matches on the line, not just the first.
    for line in text.splitlines():
        matches = re.findall(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
        for month_name_raw, day_str in matches:
            month_name = month_name_raw.title()   # Normalize: "jan" → "Jan"
            day = int(day_str)                    # "30" → 30
            month = MONTH_MAP.get(month_name)     # "Dec" → 12
            if month:
                holiday_md.add((month, day))      # (12, 30) for Dec 30

    return holiday_md
